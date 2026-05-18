# TE3002B Intelligent Robotics Challenge

Repository for coursework and mini-challenge deliverables in TE3002B — Optimal Control for Visual Servoing in Robotics.

## Course Information

- **Course:** TE3002B — Intelligent Robotics
- **Professor:** Nezih Nieto Gutiérrez

## Team

| Name | Student ID | Role |
|---|---|---|
| David Alejandro Soni Cuevas | A01571777 | Optimal Control & MPC |
| David Gilberto Lomelí Leal | A01571193 | Hardware Integration & Actuation |
| Abraham de Jesús Maldonado Mata | A00838581 | Computer Vision & Perception |

## Repository Layout

```
pzb_ros/src/
├── pzb_camera/     Camera drivers (USB cam, CSI cam, image decompressor)
├── pzb_control/    Open-loop and PID waypoint controller (Mini Challenge 2)
├── pzb_ibvs/       MPC-based Image-Based Visual Servoing (Mini Challenge 5)
├── pzb_traffic/    Traffic light detection and FSM (Mini Challenge 4)
├── pzb_utils/      Shared utilities (emergency stop, micro-ROS agent, teleop)
└── deps/           Third-party dependencies
cam_calibration/    USB camera calibration scripts and YAML outputs
```

## Packages

### `pzb_control` — Motion Controllers

Velocity and trajectory controllers for the Puzzlebot.

- **Open-loop controller** (`open_loop_controller.py`): executes YAML-defined action sequences (`move` / `turn` segments) using timed open-loop commands with reachability checks
- **Velocity controller** (`velocity_controller.py`): PID inner-loop for linear and angular velocity tracking
- **Waypoint follower** (`waypoint_follower.py`): drives the robot through a sequence of 2D waypoints using the velocity controller
- **Odometry node** (`odometry_node.py`): integrates wheel encoder data to estimate pose

```bash
# Open-loop path execution
ros2 launch pzb_control open_loop_controller.launch.py

# PID waypoint mission
ros2 launch pzb_control pid_waypoint_mission.launch.py
```

### `pzb_traffic` — Traffic Light Detection

Detects traffic light color from the camera and controls the robot with a finite-state machine.

- **Color detector** (`color_detector_node.py`): HSV-based detection of red / green / yellow
- **Traffic FSM** (`traffic_light_fsm_node.py`): stops on red, proceeds on green

```bash
# On the robot
ros2 launch pzb_traffic traffic_challenge_robot.launch.py

# PC side (visualization)
ros2 launch pzb_traffic traffic_challenge_pc.launch.py
```

### `pzb_ibvs` — MPC Image-Based Visual Servoing

Servos the Puzzlebot toward a colored target using a Model Predictive Controller driven by image features.

- **Visual detector** (`visual_detector_node.py`): color blob or ArUco marker detection
  - CLAHE normalisation, solidity filter, Gaussian pre-blur
  - Temporal holdoff to suppress single-frame dropouts
- **MPC controller** (`mpc_ibvs_node.py`):
  - 5-state augmented model: `[eu, ev, ea, v, ω]`
  - Calibrated interaction matrix with dynamic depth estimation
  - Dead zone with persistence hysteresis to prevent stutter at convergence
  - Soft-stop ramp on feature loss
  - OSQP solver with numpy fallback
- Camera: USB cam at 640×480, calibrated at RMS < 0.7 px

```bash
# Detection only (debug / tuning)
ros2 launch pzb_ibvs detect_only.launch.py

# Full MPC loop on the robot
ros2 launch pzb_ibvs mpc_ibvs_robot.launch.py

# PC side (visualization)
ros2 launch pzb_ibvs mpc_ibvs_pc.launch.py
```

## Workspace Build

```bash
cd pzb_ros
colcon build --symlink-install
source install/setup.bash
```

## Utilities

```bash
# Emergency stop (hard-stop burst)
ros2 launch pzb_utils emergency_stop.launch.py

# Start micro-ROS agent (required for MCU communication)
ros2 launch pzb_utils micro_ros_agent.launch.py

# Teleoperation with PID velocity control
ros2 launch pzb_utils teleop_pid.launch.py
```
