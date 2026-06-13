# Autonomous Lane Following and Sign Detection with Puzzlebot

TE3002B Intelligent Robotics — Manchester Robotics Challenge.

A Puzzlebot navigates a printed mini-city circuit using a single forward-facing monocular
camera as its only exteroceptive sensor. It follows lane markings on straights and curves,
detects the dashed crossing lines that mark intersections, turns left/right/straight according
to the directional sign at each junction, obeys the stop / give-way / construction signs, and
respects the traffic light. The system is a distributed ROS 2 application: the NVIDIA Jetson
Nano runs the latency-critical perception and control, while a companion laptop runs YOLOv8
sign detection on its GPU and streams the results back over the LAN.

## Course Information

- **Course:** TE3002B — Intelligent Robotics
- **Professor:** Mario Guilleremo Martinez Guerrero, Luis Ricardo Salgado

## Team

| Name | Student ID |
|---|---|
| David Alejandro Soni Cuevas | A01571777 |
| David Gilberto Lomelí Leal | A01571193 |
| Abraham de Jesús Maldonado Mata | A00838581 |

## Architecture

Two computers share one ROS 2 domain:

- **Jetson Nano (on-board):** camera capture, lane follower, traffic-light HSV detector,
  odometry, PI velocity controller, and the behavior state machine.
- **Companion laptop (GPU):** YOLOv8 sign detector accelerated with TensorRT, consuming the
  compressed camera stream and publishing detections back to the Jetson.

Motion travels through a cascade so no single node can produce an abrupt step at the motors:
the lane follower emits a desired command → a slew limiter bounds acceleration → the velocity
controller closes a PI loop on the measured body velocity → the motor-control board (ESP32 via
micro-ROS). In parallel, the behavior FSM publishes a speed scale that the follower multiplies
into its forward velocity, so a stop sign or a red light scales speed to zero without ever
perturbing the steering.

## Packages (`pzb_ros/src/`)

| Package | Responsibility |
|---|---|
| `pzb_camera` | CSI camera capture, lens undistortion, multi-stream publishing (full / hardware-downscaled / compressed JPEG). |
| `pzb_line_follower` | Python lane detection, steering control, intersection state machine, and sign-cued turns. **Primary deliverable.** |
| `pzb_line_follower_cpp` | C++ drop-in port of the follower — same node name, topics, and parameters, lower Jetson CPU/memory. |
| `pzb_traffic` | YOLOv8 sign detector, HSV traffic-light detector, and the behavior / speed-scale FSM. |
| `pzb_traffic_cpp` | C++ port of the HSV traffic-light color detector. |
| `pzb_control` | Odometry estimation, PI velocity controller, and an optional waypoint follower. |
| `pzb_utils` | Acceleration slew limiter, emergency stop, teleoperation, and the micro-ROS agent bridge. |

## Detected Signs

| Class | Meaning | Topic |
|---|---|---|
| `letIzquierda` | Left-turn arrow | `/yolo/turn_sign` |
| `letDerecha` | Right-turn arrow | `/yolo/turn_sign` |
| `letRecto` | Straight arrow | `/yolo/turn_sign` |
| `stopSign` | Stop | `/yolo/sign` |
| `GIVE WAY` | Yield | `/yolo/sign` |
| `construccion` | Construction (reduce speed) | `/yolo/sign` |

## Build

From the workspace root:

```bash
cd pzb_ros
colcon build --symlink-install
source install/setup.bash
```

## Run

The final implementation is launched from `pzb_line_follower`. On the Jetson:

```bash
# Full stack: camera + lane follower + control chain + traffic FSM (C++ follower by default)
ros2 launch pzb_line_follower line_follower.launch.py

# Use the Python follower instead of the C++ port
ros2 launch pzb_line_follower line_follower.launch.py impl:=py
```

On the companion laptop (GPU), run the YOLOv8 sign detector that publishes `/yolo/turn_sign`
and `/yolo/sign`:

```bash
ros2 launch pzb_traffic yolo_detector.launch.py
```

## Utilities

```bash
# Start the micro-ROS agent (required for MCU communication)
ros2 launch pzb_utils micro_ros_agent.launch.py

# Emergency stop
ros2 launch pzb_utils emergency_stop.launch.py

# Teleoperation with PID velocity control
ros2 launch pzb_utils teleop_pid.launch.py
```

## Repository Layout

```
pzb_ros/src/
├── pzb_camera/             CSI camera capture and image streams
├── pzb_line_follower/      Python lane follower (primary deliverable)
├── pzb_line_follower_cpp/  C++ drop-in port of the follower
├── pzb_traffic/            YOLOv8 sign detection, traffic-light FSM
├── pzb_traffic_cpp/        C++ traffic-light color detector
├── pzb_control/            Odometry, PI velocity control, waypoint follower
├── pzb_utils/              Slew limiter, e-stop, teleop, micro-ROS agent
└── deps/                   Third-party dependencies
cam_calibration/            Camera calibration scripts and YAML outputs
```
