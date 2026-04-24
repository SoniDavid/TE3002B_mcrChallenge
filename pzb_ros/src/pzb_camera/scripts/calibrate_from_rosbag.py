#!/usr/bin/env python3
"""
Offline camera calibration from a ROS2 bag using OpenCV chessboard detection.

This script reads images from a rosbag topic (raw or compressed), detects
chessboard corners, and computes intrinsics/distortion parameters with
cv2.calibrateCamera.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def parse_args():
    parser = argparse.ArgumentParser(
        description='Calibrate camera intrinsics from a ROS2 bag and chessboard images.'
    )
    parser.add_argument('--bag', required=True, help='Path to rosbag directory.')
    parser.add_argument('--topic', required=True, help='Image topic name inside the bag.')
    parser.add_argument(
        '--chessboard-size',
        nargs=2,
        type=int,
        metavar=('COLUMNS', 'ROWS'),
        required=True,
        help='Number of inner corners per chessboard row/column (e.g. 9 6).',
    )
    parser.add_argument(
        '--square-size',
        type=float,
        required=True,
        help='Chessboard square size in meters (e.g. 0.024).',
    )
    parser.add_argument(
        '--output',
        default='camera_info.yaml',
        help='Output YAML file path (ROS camera_info format).',
    )
    parser.add_argument(
        '--storage-id',
        default='sqlite3',
        help='Rosbag storage plugin id (default: sqlite3).',
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=0,
        help='Maximum images to process (0 means all).',
    )
    parser.add_argument(
        '--skip',
        type=int,
        default=1,
        help='Use every N-th message for calibration (default: 1).',
    )
    parser.add_argument(
        '--min-detections',
        type=int,
        default=15,
        help='Minimum successful chessboard detections required.',
    )
    parser.add_argument(
        '--camera-name',
        default='camera',
        help='Camera name to store in the output YAML.',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display detection preview while processing.',
    )
    return parser.parse_args()


def create_object_points(chessboard_size, square_size):
    cols, rows = chessboard_size
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * square_size
    return objp


def decode_image(msg, msg_type):
    if msg_type == 'sensor_msgs/msg/Image':
        if msg.encoding.lower() not in ('bgr8', 'rgb8', 'mono8'):
            raise ValueError(f'Unsupported Image encoding: {msg.encoding}')

        encoding = msg.encoding.lower()
        channels = 1 if encoding == 'mono8' else 3
        row_pixels = msg.width * channels

        raw = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.step > 0:
            raw = raw.reshape(msg.height, msg.step)
            raw = raw[:, :row_pixels]
        else:
            raw = raw.reshape(msg.height, row_pixels)

        if encoding == 'mono8':
            image = raw.reshape(msg.height, msg.width)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return image

        image = raw.reshape(msg.height, msg.width, 3)
        if encoding == 'rgb8':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    if msg_type == 'sensor_msgs/msg/CompressedImage':
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Failed to decode compressed image.')
        return image

    raise ValueError(f'Unsupported topic type: {msg_type}')


def get_topic_type(reader, topic_name):
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic_name not in topic_types:
        available = ', '.join(sorted(topic_types.keys()))
        raise ValueError(f'Topic {topic_name} not found. Available topics: {available}')
    return topic_types[topic_name]


def write_camera_info_yaml(path, camera_name, image_size, k, d):
    width, height = image_size

    r = np.eye(3, dtype=float)
    p = np.array(
        [
            [k[0, 0], 0.0, k[0, 2], 0.0],
            [0.0, k[1, 1], k[1, 2], 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    k_data = json.dumps(k.reshape(-1).tolist())
    d_data = json.dumps(d.reshape(-1).tolist())
    r_data = json.dumps(r.reshape(-1).tolist())
    p_data = json.dumps(p.reshape(-1).tolist())

    content = (
        f'image_width: {width}\n'
        f'image_height: {height}\n'
        f'camera_name: {camera_name}\n'
        'camera_matrix:\n'
        '  rows: 3\n'
        '  cols: 3\n'
        f'  data: {k_data}\n'
        'distortion_model: plumb_bob\n'
        'distortion_coefficients:\n'
        '  rows: 1\n'
        f'  cols: {int(d.size)}\n'
        f'  data: {d_data}\n'
        'rectification_matrix:\n'
        '  rows: 3\n'
        '  cols: 3\n'
        f'  data: {r_data}\n'
        'projection_matrix:\n'
        '  rows: 3\n'
        '  cols: 4\n'
        f'  data: {p_data}\n'
    )

    path.write_text(content, encoding='utf-8')


def main():
    args = parse_args()

    bag_path = Path(args.bag)
    if not bag_path.exists():
        raise FileNotFoundError(f'Bag path does not exist: {bag_path}')

    if args.skip < 1:
        raise ValueError('--skip must be >= 1')

    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_path),
        storage_id=args.storage_id,
    )
    converter_options = rosbag2_py.ConverterOptions('', '')

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_type = get_topic_type(reader, args.topic)
    if topic_type not in ('sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage'):
        raise ValueError(f'Topic {args.topic} has unsupported type: {topic_type}')

    msg_cls = get_message(topic_type)

    objp = create_object_points(tuple(args.chessboard_size), args.square_size)
    objpoints = []
    imgpoints = []

    message_count = 0
    used_count = 0
    image_size = None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    interrupted = False
    try:
        while reader.has_next():
            topic, data, _ = reader.read_next()
            if topic != args.topic:
                continue

            message_count += 1
            if (message_count - 1) % args.skip != 0:
                continue

            msg = deserialize_message(data, msg_cls)
            frame = decode_image(msg, topic_type)

            if image_size is None:
                image_size = (frame.shape[1], frame.shape[0])

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            found, corners = cv2.findChessboardCornersSB(gray, tuple(args.chessboard_size), None)
            if not found:
                found, corners = cv2.findChessboardCorners(
                    gray,
                    tuple(args.chessboard_size),
                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
                )
                if found:
                    corners = cv2.cornerSubPix(
                        gray,
                        corners,
                        (11, 11),
                        (-1, -1),
                        criteria,
                    )

            if found:
                objpoints.append(objp.copy())
                imgpoints.append(corners)
                used_count += 1

            if args.show:
                preview = frame.copy()
                if found:
                    cv2.drawChessboardCorners(preview, tuple(args.chessboard_size), corners, found)
                cv2.putText(
                    preview,
                    f'detections: {used_count}',
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow('calibration_preview', preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if args.max_frames > 0 and message_count >= args.max_frames:
                break
    except KeyboardInterrupt:
        interrupted = True

    if args.show:
        cv2.destroyAllWindows()

    if interrupted:
        print('Interrupted by user. Attempting calibration with collected detections...')

    if used_count < args.min_detections:
        raise RuntimeError(
            f'Not enough chessboard detections: {used_count} < {args.min_detections}. '
            'Record more views with different board positions and tilts.'
        )

    rms, k, d, _, _ = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_camera_info_yaml(output_path, args.camera_name, image_size, k, d)

    print('Calibration finished')
    print(f'  Topic:            {args.topic}')
    print(f'  Messages checked: {message_count}')
    print(f'  Detections used:  {used_count}')
    print(f'  Image size:       {image_size[0]}x{image_size[1]}')
    print(f'  RMS reprojection error: {rms:.6f}')
    print(f'  Output YAML:      {output_path}')


if __name__ == '__main__':
    main()
