# pzb_control

ROS 2 Humble package for the TE3002B Week 2 mini challenge open-loop controller.

## Package Purpose

`pzb_control` runs a path execution state machine for Puzzlebot by publishing `geometry_msgs/Twist` on `cmd_vel`.

The recommended path-definition method is YAML sequence configuration, so trajectory behavior can be modified without editing source code.

## Main Files

- Node: `scripts/mini_challenge_ctrl.py`
- Launch: `launch/mini_challenge.launch.py`
- Sequence config: `config/mini_challenge_sequence.yaml`
- Legacy parameter config: `config/mini_challenge_params.yaml`

## Sequence-Based Path Definition (Preferred)

Edit `config/mini_challenge_sequence.yaml`:

- `segment_types`: list of `move` or `turn`
- `segment_values`: list aligned by index
  - `move` value in meters
  - `turn` value in radians

Example:

```yaml
segment_types: [move, turn, move, turn, move, turn, move]
segment_values: [0.35, 1.57079632679, 0.35, 1.57079632679, 0.35, 1.57079632679, 0.35]
```

## Planning Modes

- `plan_mode: speed`
  - user sets `target_linear_speed` and `target_angular_speed`
  - controller estimates total execution time

- `plan_mode: time`
  - user sets `total_time_s` and `target_angular_speed`
  - controller estimates required linear speed

## Robustness and Safety

- Reachability checks against limits (`max_linear_speed`, `max_angular_speed`)
- Acceleration ramps (`max_linear_accel`, `max_angular_accel`)
- Minimum command magnitudes to mitigate dead-zone effects
- Runtime watchdog (`max_run_time_s`)
- Stop burst publishing (`stop_burst_cycles`)

## Build

From workspace root `pzb_ros`:

```bash
colcon build --symlink-install --packages-select pzb_control
source install/setup.bash
```

## Run

Real robot:

```bash
ros2 launch pzb_control mini_challenge.launch.py use_sim_time:=false
```

Simulation:

```bash
ros2 launch pzb_control mini_challenge.launch.py use_sim_time:=true
```

Custom parameter file:

```bash
ros2 launch pzb_control mini_challenge.launch.py params_file:=/absolute/path/to/mini_challenge_sequence.yaml
```
