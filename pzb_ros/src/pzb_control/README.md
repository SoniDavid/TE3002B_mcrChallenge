# pzb_control

ROS 2 Humble package for the Puzzlebot closed-loop velocity control stack.

## Package Purpose

`pzb_control` implements a two-layer control pipeline:

- `waypoint_follower` generates desired body velocities from waypoint error.
- `velocity_controller` closes the velocity loop using MCU feedback.
- `odometry_node` integrates feedback to produce `/odom` for the outer loop.

The parameters for all three nodes live in one YAML file so the mission, gains, and tolerances can be tuned together.

## Control Diagram

```mermaid
flowchart LR
  WPS[(waypoints_xyyaw config or /waypoints topic)] --> WF[waypoint_follower]
  ODOM_NODE[odometry_node] -->|/odom : Odometry| WF

  WF -->|/cmd_vel_desired : Twist| VC[velocity_controller]
  WF -->|/cmd_vel_desired_raw : Twist (only when use_slew_limiter:=true)| SL[twist_slew_limiter optional]
  SL -->|/cmd_vel_desired : Twist| VC
  VC -->|/cmd_vel : Twist| MCU[Robot base / MCU]
  MCU -->|/robot_vel : TwistStamped| VC
  MCU -->|/robot_vel : TwistStamped| ODOM_NODE
```

Interpretation:

- `waypoint_follower` is the outer loop. It reads pose from `/odom` and publishes a desired body twist on `/cmd_vel_desired`.
- `velocity_controller` is the inner loop. It compares `/cmd_vel_desired` with `/robot_vel`, runs PI control, and writes `/cmd_vel`.
- `odometry_node` integrates `/robot_vel` into `/odom`, and the follower uses that estimate to decide when a waypoint is reached.
- If a slew limiter is enabled, it sits between the waypoint follower and the velocity controller. In the current launch file it is defined but commented out in the returned `LaunchDescription`.

## Topic Map

| Topic | Publisher | Subscriber | Meaning |
|---|---|---|---|
| `/waypoints` | external node or mission source | `waypoint_follower` | Runtime waypoint list as `geometry_msgs/PoseArray` |
| `/odom` | `odometry_node` | `waypoint_follower` | Estimated pose used for waypoint tracking |
| `/cmd_vel_desired` | `waypoint_follower` | `velocity_controller` | Desired body velocity from the outer loop |
| `/cmd_vel` | `velocity_controller` | robot base / MCU | Corrected velocity command sent to the robot |
| `/robot_vel` | robot base / MCU | `velocity_controller`, `odometry_node` | Actual body velocity feedback from the robot |

## PID Loop Only

If you ignore the waypoint layer, the inner controller is a standard feedback loop around robot velocity.

1. A desired velocity is provided on `/cmd_vel_desired`.
2. The controller reads the actual velocity from `/robot_vel`.
3. It computes the error:

  $$e_v = v_{des} - v_{actual}$$
  $$e_w = \omega_{des} - \omega_{actual}$$

4. It runs a PI update for linear and angular velocity separately.
5. The corrected command is published on `/cmd_vel`.

In code, the controller is intentionally simple:

- proportional term reacts to current error
- integral term removes steady-state bias
- derivative is present in the class, but the YAML sets `kd_v` and `kd_w` to `0.0`

The practical effect is:

- if the robot is slower than requested, the PI controller increases command until the feedback catches up
- if the robot has a persistent bias or friction, the integral term compensates over time
- if the desired velocity becomes zero, the controller resets its integrators and publishes zero so the robot does not creep

## Waypoint Follower Plus PID

The waypoint follower is the outer loop. It does not care how the robot achieves a commanded velocity; it only decides what motion should be requested next.

The sequence is:

1. `waypoint_follower` reads the current pose from `/odom`.
2. It compares the current pose with the active waypoint.
3. It computes distance and heading error.
4. It converts that error into a desired body twist.
5. It publishes that twist on `/cmd_vel_desired`.
6. `velocity_controller` turns that request into a real `/cmd_vel` command using feedback from `/robot_vel`.

So the abstraction boundary is:

- `waypoint_follower` decides where the robot should go
- `velocity_controller` decides how to make the robot match the requested velocity

That separation is important because it lets you tune them independently:

- waypoint gains and tolerances shape path tracking behavior
- PID gains shape motor/base response and disturbance rejection

## Waypoint Logic

The waypoint follower has two phases per waypoint when yaw alignment is enabled:

- `MOVING`: drive toward the waypoint position while also steering toward the goal heading
- `ALIGNING`: once position is close enough, stop translating and rotate to the target yaw

The controller uses:

- `Kp_dist` to scale forward speed with distance error
- `Kp_yaw` to steer toward the goal heading
- `dist_tol_m` to decide when the waypoint position is reached
- `yaw_tol_rad` to decide when final heading is reached

The important behavior is that the follower only produces a desired velocity. It never directly commands the motors. The PID layer remains the only node that writes `/cmd_vel`.

## Main Files

- Node: `scripts/waypoint_follower.py`
- Node: `scripts/velocity_controller.py`
- Node: `scripts/odometry_node.py`
- Launch: `launch/pid_waypoint_mission.launch.py`
- Tuning launch: `launch/pid_velocity_ctrl_only.launch.py`
- Parameters: `config/pid_vel_params.yaml`

## Parameter File Layout

`config/pid_vel_params.yaml` is split by node name:

- `odometry_node`: pose/twist covariance and TF publishing
- `velocity_controller`: PI gains, command limits, dead-zone compensation, timeout
- `waypoint_follower`: waypoint list, tolerances, max speeds, alignment behavior

Each launch file passes the same YAML file to the relevant nodes, so the values are loaded under matching node names.

## Launch Files

### Full Mission Stack

`launch/pid_waypoint_mission.launch.py` starts:

- `odometry_node`
- `velocity_controller`
- `waypoint_follower`

It also declares an optional slew-limiter branch. The node is present in the file, but the final `LaunchDescription` currently comments it out.

### Velocity Tuning Only

`launch/pid_velocity_ctrl_only.launch.py` starts only:

- `odometry_node`
- `velocity_controller`

Use this launch file when you want to publish test commands directly to `/cmd_vel_desired` and tune the inner loop without waypoint logic.

## Message Flow

1. `waypoint_follower` reads `/odom` and computes a target body twist.
2. It publishes the target on `/cmd_vel_desired`.
3. `velocity_controller` compares desired vs actual body velocity from `/robot_vel`.
4. It publishes the corrected command to `/cmd_vel`.
5. The MCU/base executes that command and publishes new `/robot_vel` feedback.
6. `odometry_node` integrates the feedback and updates `/odom`.

## Build

From workspace root `pzb_ros`:

```bash
colcon build --symlink-install --packages-select pzb_control
source install/setup.bash
```

## Run

Full waypoint mission:

```bash
ros2 launch pzb_control pid_waypoint_mission.launch.py use_sim_time:=false
```

Velocity tuning only:

```bash
ros2 launch pzb_control pid_velocity_ctrl_only.launch.py use_sim_time:=false
```

Custom parameter file:

```bash
ros2 launch pzb_control pid_waypoint_mission.launch.py params_file:=/absolute/path/to/pid_vel_params.yaml
```
