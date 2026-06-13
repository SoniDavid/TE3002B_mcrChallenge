#!/usr/bin/env python3
"""Convert image topics in a ROS2 (SQLite3) rosbag to .mp4 video files.

Standalone — reads the .db3 directly, no ROS runtime needed. Decodes
sensor_msgs/Image (CDR) frames and writes one video per topic.

Playback is REAL-TIME by default: because the bags record at a low, variable
frame rate (~5 fps), frames are held/duplicated to a smooth fixed output FPS so a
90 s recording produces ~90 s of video (use --speed to scale). A small
"t=..s  f#.." overlay is burned into each frame so you can reference exact times.

Examples
--------
  # list image topics with dims / fps / counts
  bag_to_video.py rosbags/wide_angle_bag8 --list

  # convert the camera view + the line-follower debug strip (defaults)
  bag_to_video.py rosbags/wide_angle_bag8

  # only the raw camera, 30 fps output, 2x faster, no overlay
  bag_to_video.py rosbags/wide_angle_bag8 --topics /camera/image_raw \
                  --fps 30 --speed 2.0 --no-overlay
"""
import argparse
import glob
import os
import struct
import subprocess
import sys

import numpy as np
import cv2

DEFAULT_TOPICS = ['/camera/image_raw', '/line_follower/debug_image']
IMAGE_TYPE = 'sensor_msgs/msg/Image'

# Topics whose frames are very wide/short (e.g. the 2240x80 debug strip) are
# upscaled vertically by this factor so the resulting mp4 is watchable.
SHORT_FRAME_MAX_H = 120          # frames shorter than this get upscaled
SHORT_FRAME_UPSCALE = 3


# ── CDR sensor_msgs/Image parsing ──────────────────────────────────────────
def parse_image(data):
    """Decode a CDR-serialized sensor_msgs/Image into (h, w, encoding, ndarray).

    Layout (little-endian CDR): 4-byte encapsulation header, then
    std_msgs/Header (int32 sec, uint32 nsec, string frame_id), then
    uint32 height, uint32 width, string encoding, uint8 is_bigendian,
    uint32 step, then the uint8[] pixel sequence (uint32 length prefix).
    """
    off = 4                                   # encapsulation header
    off += 8                                  # stamp: int32 sec + uint32 nsec
    flen = struct.unpack_from('<I', data, off)[0]; off += 4 + flen   # frame_id
    off = (off + 3) & ~3                       # align to 4 for next uint32
    h, w = struct.unpack_from('<II', data, off); off += 8
    elen = struct.unpack_from('<I', data, off)[0]; off += 4
    enc = data[off:off + elen].rstrip(b'\x00').decode(); off += elen
    off += 1                                   # is_bigendian (uint8)
    off = (off + 3) & ~3
    off += 4                                   # step (uint32)
    off += 4                                   # pixel array length prefix
    # off is now the first pixel byte (==68 for the standard header in these bags)
    n = h * w * 3
    img = np.frombuffer(data, np.uint8, n, off).reshape(h, w, 3)
    return h, w, enc, img


def to_bgr(img, enc):
    """Return a writable BGR frame from a decoded image and its encoding."""
    if enc == 'bgr8':
        return img.copy()
    if enc == 'rgb8':
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    raise ValueError(f"unsupported encoding {enc!r} (only bgr8/rgb8)")


# ── bag access ─────────────────────────────────────────────────────────────
import sqlite3


def resolve_db3(path):
    if os.path.isdir(path):
        cands = sorted(glob.glob(os.path.join(path, '*_0.db3'))
                       or glob.glob(os.path.join(path, '*.db3')))
        if not cands:
            sys.exit(f"no .db3 found in {path}")
        extra = sorted(glob.glob(os.path.join(path, '*_[1-9]*.db3')))
        if extra:
            print(f"  note: bag has {len(extra)} additional split(s); only "
                  f"{os.path.basename(cands[0])} is processed", file=sys.stderr)
        return cands[0]
    return path


def image_topics(con):
    """Return {name: (topic_id, count)} for sensor_msgs/Image topics."""
    out = {}
    for tid, name, typ in con.execute("SELECT id,name,type FROM topics"):
        if typ == IMAGE_TYPE:
            n = con.execute(
                "SELECT count(*) FROM messages WHERE topic_id=?", (tid,)
            ).fetchone()[0]
            out[name] = (tid, n)
    return out


# ── overlay ────────────────────────────────────────────────────────────────
def draw_overlay(frame, rel_t, idx):
    """Burn 't=..s f#..' bottom-left, double-stroked (black under white)."""
    h = frame.shape[0]
    scale = max(0.4, min(1.0, h / 240.0))
    thick = max(1, int(round(scale * 2)))
    text = f't={rel_t:6.2f}s  f#{idx}'
    y = h - max(6, int(8 * scale))
    org = (6, y)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thick, cv2.LINE_AA)
    return frame


# ── conversion ─────────────────────────────────────────────────────────────
def _have_ffmpeg_libx264():
    try:
        out = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'],
                             capture_output=True, text=True)
        return 'libx264' in out.stdout
    except Exception:
        return False


def open_writer(path, w, h, fps):
    """Open an H.264 (libx264) ffmpeg stdin pipe — universally VLC-playable.

    Falls back to OpenCV's mp4v writer only if ffmpeg/libx264 is unavailable
    (note: mp4v = MPEG-4 Part 2, which some VLC builds, esp. on Jetson, can't
    decode — hence ffmpeg/H.264 is preferred).
    """
    if _have_ffmpeg_libx264():
        proc = subprocess.Popen(
            ['ffmpeg', '-y', '-loglevel', 'error',
             '-f', 'rawvideo', '-pix_fmt', 'bgr24',
             '-s', f'{w}x{h}', '-r', f'{fps:g}', '-i', '-',
             '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
             '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path],
            stdin=subprocess.PIPE)
        return ('ffmpeg', proc)
    print("  warn: ffmpeg/libx264 not found — falling back to OpenCV mp4v "
          "(may not play in VLC)", file=sys.stderr)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not vw.isOpened():
        sys.exit(f"could not open any video writer for {path}")
    return ('cv2', vw)


def write_frame(writer, frame):
    kind, obj = writer
    if kind == 'cv2':
        obj.write(frame)
    else:
        obj.stdin.write(frame.tobytes())


def close_writer(writer):
    kind, obj = writer
    if kind == 'cv2':
        obj.release()
    else:
        obj.stdin.close()
        obj.wait()


def convert_topic(con, name, tid, count, out_path, fps, speed, overlay):
    rows = con.execute(
        "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp",
        (tid,)).fetchall()
    if not rows:
        print(f"  skip {name}: no messages", file=sys.stderr)
        return None

    h, w, enc, _ = parse_image(rows[0][1])
    upscale = SHORT_FRAME_UPSCALE if h < SHORT_FRAME_MAX_H else 1
    out_h = h * upscale
    out_w = w

    t0 = rows[0][0]
    dt_out = speed / fps                       # virtual seconds advanced per frame
    writer = open_writer(out_path, out_w, out_h, fps)

    written = 0
    out_clock = 0.0                            # virtual bag-time of the next frame
    n = len(rows)
    for i, (ts, data) in enumerate(rows):
        rel = (ts - t0) / 1e9
        try:
            _, _, e, img = parse_image(data)
            frame = to_bgr(img, e)
        except Exception as ex:                # corrupt frame → hold previous
            print(f"  warn {name} f#{i}: decode failed ({ex})", file=sys.stderr)
            continue
        if upscale != 1:
            frame = cv2.resize(frame, (out_w, out_h),
                               interpolation=cv2.INTER_NEAREST)
        # Hold this frame until the output clock reaches the NEXT message's time
        # (or, for the last frame, write it once). Duplicates fill real-time gaps.
        next_rel = (rows[i + 1][0] - t0) / 1e9 if i + 1 < n else rel + dt_out
        # always emit at least one frame per message
        emitted_this_msg = False
        while out_clock <= next_rel or not emitted_this_msg:
            f = frame if not overlay else draw_overlay(frame.copy(), rel, i)
            write_frame(writer, f)
            written += 1
            out_clock += dt_out
            emitted_this_msg = True
            if i + 1 >= n:                     # last message: one frame only
                break

    close_writer(writer)
    real_dur = (rows[-1][0] - t0) / 1e9
    vid_dur = written / fps
    print(f"  {name}")
    print(f"    -> {out_path}")
    print(f"       {n} msgs, {w}x{h} {enc}"
          f"{f' (upscaled {upscale}x -> {out_w}x{out_h})' if upscale!=1 else ''}")
    print(f"       wrote {written} frames @ {fps}fps = {vid_dur:.1f}s video "
          f"(bag {real_dur:.1f}s, speed {speed}x)")
    return out_path


def sanitize(topic):
    return topic.strip('/').replace('/', '_')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bag', help='bag directory or .db3 file')
    ap.add_argument('--topics', nargs='+', default=None,
                    help=f'image topics (default: {" ".join(DEFAULT_TOPICS)})')
    ap.add_argument('--fps', type=float, default=25.0, help='output fps (default 25)')
    ap.add_argument('--speed', type=float, default=1.0,
                    help='playback speed (1.0=real time, default 1.0)')
    ap.add_argument('--out-dir', default=None,
                    help='output directory (default: next to the bag)')
    ap.add_argument('--no-overlay', action='store_true',
                    help='do not burn timestamp/frame# into frames')
    ap.add_argument('--list', action='store_true',
                    help='list image topics and exit')
    args = ap.parse_args()

    db3 = resolve_db3(args.bag)
    con = sqlite3.connect(db3)
    topics = image_topics(con)

    if args.list:
        print(f"Image topics in {db3}:")
        for name, (tid, n) in topics.items():
            row = con.execute(
                "SELECT timestamp,data FROM messages WHERE topic_id=? "
                "ORDER BY timestamp LIMIT 1", (tid,)).fetchone()
            if not row:
                print(f"  {name:34} (empty)")
                continue
            last = con.execute(
                "SELECT timestamp FROM messages WHERE topic_id=? "
                "ORDER BY timestamp DESC LIMIT 1", (tid,)).fetchone()[0]
            dur = (last - row[0]) / 1e9
            fps = (n - 1) / dur if dur > 0 else 0
            try:
                h, w, enc, _ = parse_image(row[1])
                dims = f'{w}x{h} {enc}'
            except Exception as ex:
                dims = f'(parse failed: {ex})'
            print(f"  {name:34} n={n:<5} ~{fps:4.1f}fps  {dims}")
        return

    bag_name = os.path.basename(os.path.dirname(db3) if os.path.isdir(args.bag)
                                else db3).replace('.db3', '')
    if os.path.isdir(args.bag):
        bag_name = os.path.basename(os.path.normpath(args.bag))
    out_dir = args.out_dir or (args.bag if os.path.isdir(args.bag)
                               else os.path.dirname(db3))
    os.makedirs(out_dir, exist_ok=True)

    want = args.topics or DEFAULT_TOPICS
    produced = []
    print(f"Converting {db3}  (fps={args.fps}, speed={args.speed}x, "
          f"overlay={'off' if args.no_overlay else 'on'})")
    for name in want:
        if name not in topics:
            print(f"  skip {name}: not an image topic in this bag", file=sys.stderr)
            continue
        tid, count = topics[name]
        out_path = os.path.join(out_dir, f'{bag_name}__{sanitize(name)}.mp4')
        p = convert_topic(con, name, tid, count, out_path,
                          args.fps, args.speed, not args.no_overlay)
        if p:
            produced.append(p)

    if produced:
        print(f"\nDone. {len(produced)} video(s) written.")
    else:
        print("\nNo videos written.", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
