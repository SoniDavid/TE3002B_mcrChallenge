#!/usr/bin/env bash
# Start the OFF-BOARD YOLO sign/traffic detector on the LAPTOP (GPU).
#
# WHY: the YOLO model runs on the laptop, never on the Jetson. It subscribes to the
# Jetson's /camera/image_compressed (JPEG) and publishes /yolo/sign ("class:area")
# for the line follower. This wraps the ros2 launch so you don't retype the model
# path each time, and so the path isn't hardcoded to one machine.
#
# Usage:
#   YOLO_MODEL=/abs/path/to/best.pt ./scripts/run_yolo.sh
#   ./scripts/run_yolo.sh /abs/path/to/best.pt          # path as 1st arg
#   YOLO_MODEL=... CONF=0.5 DEVICE=cpu ./scripts/run_yolo.sh
#
# Env / args:
#   YOLO_MODEL (or $1)  absolute path to the .pt weights (required)
#   CONF                detection confidence threshold (default 0.55)
#   DEVICE              cuda | cpu (default cuda)

set -euo pipefail

MODEL="${1:-${YOLO_MODEL:-}}"
CONF="${CONF:-0.55}"
DEVICE="${DEVICE:-cuda}"

if [[ -z "${MODEL}" ]]; then
  echo "ERROR: no model path. Set YOLO_MODEL=/abs/path/best.pt or pass it as the first arg." >&2
  echo "  e.g. YOLO_MODEL=~/best.pt ./scripts/run_yolo.sh" >&2
  exit 1
fi
if [[ ! -f "${MODEL}" ]]; then
  echo "ERROR: model not found: ${MODEL}" >&2
  exit 1
fi

echo "Launching YOLO detector  model=${MODEL}  conf=${CONF}  device=${DEVICE}"
exec ros2 launch pzb_traffic yolo_detector.launch.py \
  model_path:="${MODEL}" \
  conf_threshold:="${CONF}" \
  device:="${DEVICE}"
