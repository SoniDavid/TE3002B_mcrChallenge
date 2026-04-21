#!/bin/bash
# Installs Arducam ISP tuning file to fix IMX219 red tint on Jetson.
# Run once after flashing or if the red tint returns after a reboot.

set -e

SETTINGS_DIR="/var/nvidia/nvcam/settings"
ISP_FILE="camera_overrides.isp"
ARCHIVE="Camera_overrides.tar.gz"
TMPDIR="$(mktemp -d)"

cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

cd "$TMPDIR"

echo "[1/5] Downloading ISP tuning file..."
if ! wget -q --show-progress https://www.arducam.com/downloads/Jetson/Camera_overrides.tar.gz -O "$ARCHIVE"; then
    echo "Arducam mirror failed, trying Waveshare..."
    wget -q --show-progress https://files.waveshare.com/upload/e/eb/Camera_overrides.tar.gz -O "$ARCHIVE" || {
        echo "Waveshare mirror failed, trying GitHub raw file..."
        wget -q --show-progress \
            https://raw.githubusercontent.com/sfalexrog/jetson_camera/master/extra/camera_overrides.isp \
            -O "$ISP_FILE"
        ARCHIVE=""
    }
fi

if [ -n "$ARCHIVE" ]; then
    echo "[2/5] Extracting archive..."
    tar zxf "$ARCHIVE"
else
    echo "[2/5] Skipping extract (raw .isp downloaded directly)"
fi

echo "[3/5] Clearing ISP cache..."
sudo rm -f "$SETTINGS_DIR"/nvcam_cache_* "$SETTINGS_DIR"/serial_no_*

echo "[4/5] Installing ISP file..."
sudo cp "$ISP_FILE" "$SETTINGS_DIR/"
sudo chmod 664 "$SETTINGS_DIR/$ISP_FILE"
sudo chown root:root "$SETTINGS_DIR/$ISP_FILE"

echo "[5/5] Restarting nvargus-daemon..."
sudo systemctl restart nvargus-daemon

echo ""
echo "Done. Test with:"
echo "  gst-launch-1.0 nvarguscamerasrc num-buffers=1 ! \\"
echo "    'video/x-raw(memory:NVMM), width=1280, height=720' ! \\"
echo "    nvvidconv ! jpegenc ! filesink location=~/test_color.jpg"
