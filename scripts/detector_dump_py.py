#!/usr/bin/env python3
"""Phase-A parity reference: run the PYTHON CenterLineDetector over a bag's
/camera/image_small with injected bag-timestamp dt (same as replay_follower.py) and dump
per-frame t,cx,cy,line_type,n_vis,prev_cx as CSV — to diff against the C++ detector_dump.

Usage: detector_dump_py.py <bag_dir> [--topic /camera/image_small] [--csv out.csv]
"""
import argparse, glob, os, sqlite3, sys, types
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'pzb_ros', 'src', 'pzb_line_follower', 'scripts'))
sys.path.insert(0, HERE)
from bag_to_video import parse_image, to_bgr
import center_line_detector as cld_mod
from center_line_detector import CenterLineDetector


def crop_resize_like_node(full, img_w=320, img_h=240):
    roi = full[full.shape[0] // 2:, :]
    th, tw = img_h // 3, img_w
    if roi.shape[0] != th or roi.shape[1] != tw:
        return cv2.resize(roi, (tw, th), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(roi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('--topic', default='/camera/image_small')
    ap.add_argument('--csv', default=None)
    args = ap.parse_args()

    db = sorted(glob.glob(os.path.join(args.bag, '*_0.db3')) or
                glob.glob(os.path.join(args.bag, '*.db3')))[0]
    con = sqlite3.connect(db); cur = con.cursor()
    tid = {n: i for i, n in cur.execute("SELECT id,name FROM topics")}[args.topic]
    rows = cur.execute("SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp",
                       (tid,)).fetchall()

    bag_now = {'t': 0.0}
    cld_mod.time = types.SimpleNamespace(monotonic=lambda: bag_now['t'])
    det = CenterLineDetector(debug=False)

    out = open(args.csv, 'w') if args.csv else sys.stdout
    out.write("t,cx,cy,line_type,n_vis,prev_cx\n")
    t0 = rows[0][0]
    n = 0
    for ts, data in rows:
        bag_now['t'] = (ts - t0) / 1e9
        try:
            h, w, e, img = parse_image(data); img = to_bgr(img, e)
        except Exception:
            continue
        roi = crop_resize_like_node(img)
        cx, cy = det.detect_center_line(roi, pre_cropped=True)
        nvis = sum(det.line_flags.values())
        pc = det.prev_cx if det.prev_cx is not None else float('nan')
        out.write(f"{bag_now['t']:.6f},{cx},{cy},{det.line_type},{nvis},{pc:.1f}\n")
        n += 1
    if args.csv:
        out.close()
    sys.stderr.write(f"frames: {n}\n")


if __name__ == '__main__':
    main()
