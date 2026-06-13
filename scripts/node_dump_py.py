#!/usr/bin/env python3
"""Drive the REAL current line_follower_node steering over a bag's /camera/image_small with
injected bag-timestamp dt, and dump per-frame cx/error/line_type/v/w. Apples-to-apples
reference for the C++ FollowerCore (both run the SAME current logic). rclpy is stubbed so
the node instantiates without a ROS runtime.

Usage: node_dump_py.py <bag> [--csv out.csv]
"""
import argparse, glob, os, sqlite3, sys, types
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
LF = os.path.join(ROOT, 'pzb_ros', 'src', 'pzb_line_follower', 'scripts')
sys.path.insert(0, LF); sys.path.insert(0, HERE)
from bag_to_video import parse_image, to_bgr

# ── stub rclpy + msgs so line_follower_node imports/instantiates ──
fake_t = {'t': 0.0}
import time as _time
_time.monotonic = lambda: fake_t['t']

rclpy = types.ModuleType('rclpy'); rclpy.init = lambda *a, **k: None
rclpy.spin = lambda *a, **k: None; rclpy.try_shutdown = lambda *a, **k: None
nodemod = types.ModuleType('rclpy.node')
qos = types.ModuleType('rclpy.qos')
class _QoSProfile:
    def __init__(s,*a,**k): pass
qos.QoSProfile = _QoSProfile
qos.ReliabilityPolicy = types.SimpleNamespace(BEST_EFFORT=1, RELIABLE=2, SYSTEM_DEFAULT=0)
qos.DurabilityPolicy  = types.SimpleNamespace(VOLATILE=1, TRANSIENT_LOCAL=2, SYSTEM_DEFAULT=0)
qos.HistoryPolicy     = types.SimpleNamespace(KEEP_LAST=1, KEEP_ALL=2, SYSTEM_DEFAULT=0)
class _Pub:
    def publish(self, m): pass
class FakeNode:
    def __init__(s,*a,**k): pass
    def declare_parameter(s,n,d): s.__dict__.setdefault('_p',{})[n]=d; return types.SimpleNamespace(value=d)
    def get_parameter(s,n): return types.SimpleNamespace(value=s._p[n])
    def create_publisher(s,*a,**k): return _Pub()
    def create_subscription(s,*a,**k): return None
    def create_timer(s,*a,**k): return None
    def get_logger(s):
        return types.SimpleNamespace(info=lambda *a,**k:None, warn=lambda *a,**k:None,
                                     warning=lambda *a,**k:None, error=lambda *a,**k:None)
    def get_clock(s):
        return types.SimpleNamespace(now=lambda: types.SimpleNamespace(
            to_msg=lambda: types.SimpleNamespace(sec=0, nanosec=0)))
nodemod.Node = FakeNode
rclpy.node = nodemod; rclpy.qos = qos
sys.modules['rclpy']=rclpy; sys.modules['rclpy.node']=nodemod; sys.modules['rclpy.qos']=qos

# The node imports the detector as pzb_line_follower_scripts.center_line_detector
# (the installed package path); alias it to the source module.
import center_line_detector as _cld
_pkg = types.ModuleType('pzb_line_follower_scripts')
sys.modules['pzb_line_follower_scripts'] = _pkg
sys.modules['pzb_line_follower_scripts.center_line_detector'] = _cld

import line_follower_node as lfn


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('bag')
    ap.add_argument('--csv', default=None); args = ap.parse_args()
    db = sorted(glob.glob(os.path.join(args.bag,'*_0.db3')) or glob.glob(os.path.join(args.bag,'*.db3')))[0]
    con = sqlite3.connect(db); cur = con.cursor()
    tid = {n:i for i,n in cur.execute("SELECT id,name FROM topics")}['/camera/image_small']
    rows = cur.execute("SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp",(tid,)).fetchall()

    node = lfn.LineFollowerNode()
    node._pub_debug = False   # skip debug-image build (no real ROS msgs in this harness)
    # capture published cx/error/line_type by wrapping the publishers
    cap = {'cx':0,'err':0.0,'lt':'solid'}
    node._pub_cx.publish    = lambda m: cap.__setitem__('cx', m.data)
    node._pub_error.publish = lambda m: cap.__setitem__('err', m.data)
    node._pub_line_type.publish = lambda m: cap.__setitem__('lt', m.data)

    out = open(args.csv,'w') if args.csv else sys.stdout
    out.write("t,cx,err,line_type,v,w\n")
    t0 = rows[0][0]; n=0
    for ts,data in rows:
        fake_t['t'] = (ts - t0)/1e9
        try:
            h,w,e,img = parse_image(data); img = to_bgr(img,e)
        except Exception:
            continue
        msg = types.SimpleNamespace(
            data=img.tobytes(), height=img.shape[0], width=img.shape[1],
            header=types.SimpleNamespace(stamp=None, frame_id=''))
        node._image_cb(msg)
        cmd = node._latest_cmd
        out.write(f"{fake_t['t']:.6f},{cap['cx']},{cap['err']:.1f},{cap['lt']},{cmd.linear.x:.4f},{cmd.angular.z:.4f}\n")
        n+=1
    if args.csv: out.close()
    sys.stderr.write(f"frames: {n}\n")


if __name__ == '__main__':
    main()
