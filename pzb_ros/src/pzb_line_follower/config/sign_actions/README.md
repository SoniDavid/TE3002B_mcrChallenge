# Teach-by-demonstration sign actions (ROUND 9)

Each `<sign>.csv` is a recorded `/cmd_vel` maneuver replayed OPEN-LOOP when that turn sign
is latched at a dashed crossing. Signs: `left.csv`, `right.csv`, `straight.csv`
(letIzquierda / letDerecha / letRecto).

Format — one header line then `t,vx,wz` rows:

```
t,vx,wz
0.0,0.05,0.0
0.12,0.05,0.45
...
```

- `t`  = seconds from the start of the action (held-last-value until the next row)
- `vx` = linear.x (m/s), `wz` = angular.z (rad/s)

## How to (re)generate
1. Bring up camera + traffic only (teleop owns /cmd_vel):
   `ros2 launch pzb_line_follower line_follower.launch.py launch_follower:=false launch_control:=false`
2. Run YOLO on the laptop + `teleop_twist_keyboard`.
3. Record the correct maneuver (start before the dashed line, do the turn, settle):
   `cd pzb_ros/scripts && ./record_run.sh left_action`   (etc. for right/straight)
4. Extract, aligning the action start to the dashed crossing:
   `scripts/extract_sign_action.py pzb_ros/scripts/rosbags/left_action left --from-dashed`
5. Rebuild `pzb_line_follower` (+ the cpp pkg) and relaunch.

If a CSV is absent the follower falls back to the synthetic cross_turn arc for that sign.
