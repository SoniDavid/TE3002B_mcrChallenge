# pzb_camera

ROS 2 Humble package for the IMX219 CSI camera on Jetson Nano.

Publishes compressed JPEG frames over `/camera/image_compressed` for network transmission, and raw frames over `/camera/image_raw` for local processing nodes.

## Main Files

- Node: `scripts/camera_publisher.py`
- Launch: `launch/camera.launch.py`
- Config: `config/camera_params.yaml`

## Published Topics

| Topic | Type | Description |
|---|---|---|
| `/camera/image_compressed` | `sensor_msgs/CompressedImage` | JPEG-compressed frames. Use this over the network. |
| `/camera/image_raw` | `sensor_msgs/Image` | Raw BGR frames. Only published when a subscriber is active. |

## Supported Resolutions (IMX219)

| Width | Height | Max FPS |
|---|---|---|
| 3264 | 2464 | 21 |
| 3264 | 1848 | 28 |
| 1920 | 1080 | 30 |
| 1640 | 1232 | 30 |
| 1280 | 720 | 60 |

## Prerequisites

### 1. Install system dependencies

```bash
sudo apt update
sudo apt install python3-opencv
```

### 2. Install ROS 2 Python dependencies

```bash
sudo apt install ros-humble-cv-bridge ros-humble-sensor-msgs
```

### 3. Start the NVIDIA Argus camera daemon

This must be running before opening the camera. Check once at boot:

```bash
sudo systemctl start nvargus-daemon

# Verify it is active
sudo systemctl status nvargus-daemon
```

To start it automatically on boot:

```bash
sudo systemctl enable nvargus-daemon
```

### 4. Verify the camera is detected

```bash
ls /dev/video0
v4l2-ctl -d /dev/video0 --info
```

The card type should show `imx219`.

## Build

From workspace root `pzb_ros`:

```bash
colcon build --symlink-install --packages-select pzb_camera
source install/setup.bash
```

## Run

Default settings (1280x720 @ 30 fps, JPEG quality 80):

```bash
ros2 launch pzb_camera camera.launch.py
```

Override parameters inline:

```bash
ros2 launch pzb_camera camera.launch.py width:=1920 height:=1080 framerate:=30 jpeg_quality:=60
```

Custom parameter file:

```bash
ros2 launch pzb_camera camera.launch.py params_file:=/absolute/path/to/camera_params.yaml
```

Direct node run (no launch file):

```bash
ros2 run pzb_camera camera_publisher --ros-args --params-file install/pzb_camera/share/pzb_camera/config/camera_params.yaml
```

## Verify Output

```bash
# List active topics
ros2 topic list | grep camera

# Check publishing rate
ros2 topic hz /camera/image_compressed

# Check bandwidth
ros2 topic bw /camera/image_compressed

# Inspect a single message (shows header and format)
ros2 topic echo --once /camera/image_compressed
```

## View the Stream

On a remote machine in the same ROS 2 network:

```bash
ros2 run rqt_image_view rqt_image_view /camera/image_compressed
```

Or with rviz2: add an `Image` display and set topic to `/camera/image_compressed`.

## Troubleshooting

**`Failed to open camera`**

- Check nvargus-daemon is running: `sudo systemctl start nvargus-daemon`
- Re-seat the CSI flex cable (both ends, lock tabs closed, blue strip facing heatsink)
- Confirm no other process has the camera open: `sudo fuser /dev/video0`

**`Could not open display`** error from GStreamer

- This is expected when running over SSH with no display attached. The ROS node does not need a display.

**Low framerate or frame drops**

- Reduce `jpeg_quality` (try 60) or lower resolution in `camera_params.yaml`
- Check CPU load: `jtop` or `htop`

**`ImportError: cv2`**

- Install OpenCV: `sudo apt install python3-opencv`

**`ModuleNotFoundError: cv_bridge`**

- Install cv_bridge: `sudo apt install ros-humble-cv-bridge`
