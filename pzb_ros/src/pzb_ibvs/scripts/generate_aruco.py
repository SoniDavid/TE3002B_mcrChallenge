#!/usr/bin/env python3
"""
Generate an ArUco marker image ready to display or print.

Usage:
    python3 generate_aruco.py                   # ID 0, DICT_4X4_50, 600px
    python3 generate_aruco.py --id 3            # different ID
    python3 generate_aruco.py --size 900        # larger image
    python3 generate_aruco.py --out my_marker.png
"""

import argparse
import cv2
import numpy as np


DICT_MAP = {
    'DICT_4X4_50':   cv2.aruco.DICT_4X4_50,
    'DICT_4X4_100':  cv2.aruco.DICT_4X4_100,
    'DICT_5X5_50':   cv2.aruco.DICT_5X5_50,
    'DICT_6X6_50':   cv2.aruco.DICT_6X6_50,
}


def main():
    parser = argparse.ArgumentParser(description='Generate an ArUco marker PNG.')
    parser.add_argument('--id',   type=int,   default=0,              help='Marker ID (default: 0)')
    parser.add_argument('--dict', type=str,   default='DICT_4X4_50',  help='ArUco dictionary (default: DICT_4X4_50)')
    parser.add_argument('--size', type=int,   default=600,            help='Output image size in pixels (default: 600)')
    parser.add_argument('--out',  type=str,   default='',             help='Output filename (default: aruco_<dict>_id<id>.png)')
    parser.add_argument('--border', type=int, default=1,              help='White border width in cells (default: 1)')
    args = parser.parse_args()

    if args.dict not in DICT_MAP:
        print(f'Unknown dict "{args.dict}". Choose from: {list(DICT_MAP.keys())}')
        return

    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_MAP[args.dict])
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, args.id, args.size)

    # Add a white border so the detector can find the outer edge
    border_px = max(1, args.size // 10)
    marker_with_border = cv2.copyMakeBorder(
        marker_img,
        border_px, border_px, border_px, border_px,
        cv2.BORDER_CONSTANT, value=255,
    )

    out_file = args.out or f'aruco_{args.dict}_id{args.id}.png'
    cv2.imwrite(out_file, marker_with_border)

    total_px = args.size + 2 * border_px
    print(f'Saved: {out_file}  ({total_px}x{total_px} px)')
    print(f'  Dictionary : {args.dict}')
    print(f'  Marker ID  : {args.id}')
    print()
    print('Display this image fullscreen on your monitor/phone.')
    print('Make sure the white border is fully visible — the detector needs it.')
    print()
    print('Config values to use in mpc_ibvs_params.yaml:')
    print(f'  aruco_dict : {args.dict}')
    print(f'  marker_id  : {args.id}')


if __name__ == '__main__':
    main()
