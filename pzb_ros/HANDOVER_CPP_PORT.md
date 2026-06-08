# Handover — C++ port of the line follower + color detector (`feature/cpp-port`)

## TL;DR
The resource-heavy nodes were ported to **C++/ROS 2** to cut Jetson CPU/memory, kept
**parallel** to the Python (Python untouched). Two new ament_cmake packages —
`pzb_line_follower_cpp` and `pzb_traffic_cpp` — are **drop-in**: same node names, topics,
and parameters. Select the implementation with a launch arg:

```bash
ros2 launch pzb_line_follower line_follower.launch.py impl:=cpp   # C++ follower
ros2 launch pzb_line_follower line_follower.launch.py impl:=py    # Python (default)
```

Offline-validated against the rosbags: the C++ detector matches the Python **100% within
1px**; the full C++ node matches the current Python node **98.8% within 1px**.

This branch ALSO contains two behavior changes (Python is the spec, then re-ported to
C++): **dashed alignment re-enabled** and **crossing-state held across the
inter-intersection gap**. See "Behavior changes" below.

Branch: `feature/cpp-port` (off `dev`). Commits: `c52fd92` (A), `6e1f6f3` (B), `b0b4e52`
(C), `7336468` (param fix). **Not merged to dev. Commits author as the user — do NOT add
Claude as co-author.**

---

## What was ported (and what was NOT)
PORTED to C++:
- `center_line_detector.py`  → `pzb_line_follower_cpp` detector library.
- `line_follower_node.py`     → `pzb_line_follower_cpp` (full steering state machine).
- `color_detector_node.py`    → `pzb_traffic_cpp`.

NOT ported (intentional — negligible CPU or already C-backed):
- `traffic_light_fsm_node.py` (pure logic/timers) — stays Python.
- `camera_raw_publisher.py`   (GStreamer/VIC hardware, already C-backed) — stays Python.

## Package layout (`pzb_ros/src/pzb_line_follower_cpp/`)
- `include/pzb_line_follower_cpp/center_line_detector.hpp` + `src/center_line_detector.cpp`
  — detector class. All Python class constants are `static constexpr`; algorithm order is
  1:1 with the Python.
- `include/.../follower_core.hpp` + `src/follower_core.cpp` — **ROS-free** steering state
  machine (`FollowerParams` struct + `FollowerCore::process_frame`). The PD path, curve_gain,
  turn-latch (B2), dashed align + crossing-gap, search/stuck-lock/error-sat/stale, and YOLO
  turn-sign gating all live here so the offline comparator can drive the exact same logic.
- `src/line_follower_node.cpp` — thin rclcpp wrapper around `FollowerCore`. Declares all
  ~60 params, BEST_EFFORT image sub, RELIABLE cx/error/line_type/cmd pubs, 20 Hz publish
  timer with the stale/blind-frame safety.
- `include/.../bag_reader.hpp` + `src/bag_reader.cpp` — ROS-free rosbag (`sqlite3`) reader +
  CDR `sensor_msgs/Image|Int32|Float32|String` parse (mirrors `scripts/bag_to_video.py`).
- `src/detector_dump_main.cpp`   → `detector_dump`  executable (Phase-A detector parity).
- `src/replay_follower_main.cpp` → `replay_follower` executable (Phase-B node comparator).

`pzb_traffic_cpp/src/color_detector_node.cpp` — standalone rclcpp HSV detector.

## Drop-in mechanism
The C++ nodes run under the SAME node names (`line_follower_node`, `color_detector_node`)
and read the SAME YAML (`pzb_line_follower/config/line_follower_params.yaml`,
`pzb_traffic/config/traffic_params.yaml`). `line_follower.launch.py` has an `impl:=py|cpp`
arg (default `py`) that swaps only the package via a `PythonExpression`. Everything
downstream (twist_slew_limiter, traffic FSM, bags) is unchanged.

## Faithful-port rules (so the bag comparison matches — keep these)
- **Build WITHOUT `-ffast-math`** (CMake sets `-O2` only). FP order-of-ops already causes
  occasional ±1px differences at `int(round())` boundaries; fast-math would worsen it.
- **Even-length median = mean of the two middle values** (numpy.median). `error_median_n=2`
  is even, so a `.5` is possible — `median_even()` handles it. Detector medians are maxlen-3
  (odd → middle, then int-cast/truncate).
- **Adaptive threshold** (`adaptive_threshold`) is a frame-to-frame feedback loop (carries
  `T_state_`, ±10 steps). Ported bit-for-bit — this is the #1 divergence amplifier; a single
  different `T` cascades. Do not "simplify" it.
- **Injectable clock**: every time source goes through a `std::function<double()>`. The
  offline tools feed the **bag timestamp** (matching `replay_follower.py`'s monkeypatch); the
  rclcpp node uses the **image header stamp** so dt matches a replayed run. Default =
  `steady_clock`.
- **`resolve_db3` uses POSIX `dirent`, NOT `std::filesystem`** — `std::filesystem`
  destructors SEGFAULT on this Jetson's GCC toolchain (learned the hard way).
- OpenCV is 4.2 for both `cv2` and C++ (same core), so `findContours` ordering matches.
- **Parameter types**: the YAML writes many `*_px`/`align_*` doubles as bare integers
  (`stuck_lock_band_px: 20`). rclcpp refuses an int→double override
  (`InvalidParameterTypeException`). The node uses a `gdf()` helper that declares such
  params with `dynamic_typing` and coerces int→double. **If you add a new double param that
  may have an integer YAML literal, declare it with `gdf`, not `gd`.**

## Validation (how it was checked, how to re-run)
Apples-to-apples = both run the SAME current logic with bag-injected dt. Comparing C++ to
the values RECORDED in old bags is NOT valid parity (those predate the lean rewrite +
align/gap changes).
- **Detector**: `scripts/detector_dump_py.py <bag>` (Python) vs
  `detector_dump <bag>` (C++) → diff cx/line_type/n_vis. Result: 100% within 1px, 0%
  line_type/n_vis mismatch.
- **Full node**: `scripts/node_dump_py.py <bag>` drives the REAL current Python node (rclpy
  stubbed) → cx/err/line_type/v/w; `replay_follower <bag>` drives the C++ `FollowerCore`.
  Result on `pzb_no_turning_well2`: **98.8% within 1px, mean |Δcx|=0.74, line_type mismatch
  2.3%**. The residual is a ~6-frame slot-swap excursion at an ambiguous multi-line frame
  that self-corrects (FP tie-break in `best_assignment` + single-boundary lane-center bias).
- Color detector: NOT bag-validatable here — the available bags have no `/traffic_light_color`
  and no light in view. Verify live.

## Behavior changes on this branch (Python = spec, mirrored in C++)
1. **Dashed alignment re-enabled** (`dashed_align_enabled: true` in the YAML). The robot
   squares up perpendicular to the dash row before crossing/turning (real `_alignment_cmd`
   path, using the median dash-slope + `roi_aniso=0.889`).
2. **Crossing-state ownership across the inter-intersection gap.** Once a dashed crossing is
   confirmed, the crossing FSM keeps control (align → YOLO turn/straight → **coast** at
   `crossing_coast_speed`) and does NOT revert to line-following in the gap (there is no real
   line there, so the detector's `(L+R)/2` fallback would steer to noise). It exits only when
   a real lane returns: ≥2 line slots for `crossing_exit_frames` (=3) consecutive frames →
   reset anchors → normal centering. Turn-left = open-loop arc then re-center on the new lane.
   A false dashed-on-curve (live-error break) releases crossing-state immediately.
   New params: `crossing_coast_speed`, `crossing_exit_frames`. State (Python):
   `_crossing_active`, `_crossing_done_t`, `_real_line_streak` (C++: `crossing_active_`,
   `crossing_done_t_`, `real_line_streak_`).

## Known non-issues (warnings you'll see)
- `Robot interface missing: /cmd_vel subscribers=0, /robot_vel publishers=0` — the
  velocity_controller noting the micro-ROS/MCU bridge isn't running. Expected without the base.
- `camera_raw_publisher: ... publisher's context is invalid` **at Ctrl-C only** — a benign
  shutdown race: the GStreamer publisher worker threads (from the stall fix) try to publish
  after the rclcpp context tears down. Cosmetic; fix by stopping the workers before context
  shutdown if it bothers you.

## TODO / next steps
- **On-Jetson A/B (the real payoff):** `colcon build` the branch, then run `impl:=cpp` vs
  `impl:=py` with `jtop` open — compare CPU% / memory and `/line_follower/cx` fps. Confirm
  the C++ matches behavior live.
- Decide whether to merge to `dev` or keep parallel.
- Optional: clean up the camera shutdown race; bag-validate the color detector with a light.

## Files
- C++: `pzb_ros/src/pzb_line_follower_cpp/`, `pzb_ros/src/pzb_traffic_cpp/`
- Launch arg: `pzb_ros/src/pzb_line_follower/launch/line_follower.launch.py` (`impl`)
- Behavior change (Python): `pzb_ros/src/pzb_line_follower/scripts/line_follower_node.py`,
  `pzb_ros/src/pzb_line_follower/config/line_follower_params.yaml`
- Validation tooling: `scripts/detector_dump_py.py`, `scripts/node_dump_py.py`,
  `scripts/replay_follower.py`, `scripts/bag_to_video.py`, and the C++ `detector_dump` /
  `replay_follower` executables.
