# TE3002B Intelligent Robotics Challenge

Repository for coursework and practical challenges in TE3002B Intelligent Robotics.

## Course Information

- Course: TE3002B - Intelligent Robotics - Manchester Robotics Puzzlebot Module
- Professor: Dr. Mario Martinez (Or I don't actually know who checks this in Campus MTY lol)

### Students

| Name | Student ID |
|---|---|
| David Alejandro Soni Cuevas | A01571777 |


## Featured Project: Week 2 Challenge

### Puzzlebot Open-Loop Path Execution

Goal: execute user-defined trajectories on Puzzlebot using open-loop control with robust safety behavior.

Core approach:

- YAML-defined action sequence (`move` and `turn` segments)
- Open-loop timing model using commanded velocities
- Reachability checks against dynamic limits
- State-machine segment execution
- Emergency stop utility node for hard stop bursts

Detailed package documentation is available in:

- [pzb_ros/src/pzb_control/README.md](pzb_ros/src/pzb_control/README.md)

## Workspace Quick Start

### Repository Layout

- `pzb_ros/src/pzb_control`: challenge controller package
- `pzb_ros/src/pzb_utils`: utility package (emergency stop)

### Build

```bash
cd pzb_ros
colcon build --symlink-install
source install/setup.bash
```

### Run Challenge Controller

```bash
ros2 launch pzb_control mini_challenge.launch.py use_sim_time:=false
```

### Emergency Stop

```bash
ros2 run pzb_utils emergency_stop
```
