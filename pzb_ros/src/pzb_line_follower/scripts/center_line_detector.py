from collections import deque
from itertools import combinations
import numpy as np
import cv2


class CenterLineDetector:
    """
    v4 — three-line tracker + dashed-line (intersection) detection.

    Solid mode
    ──────────
    Tracks left / center / right lines independently:
      - Each line has its own median-filter history and prev_cx anchor.
      - Contours are assigned to lines via minimum-cost matching so markers
        never swap identities between frames.
      - A per-line velocity gate rejects jumps larger than LINE_MAX_JUMP px
        (the cart doesn't teleport).
      - Ordering is enforced: left_cx < center_cx < right_cx with at least
        LINE_MIN_SEP px between neighbours.  Violations flag all lines lost.

    Public attributes updated each call
    ────────────────────────────────────
    line_type      : "solid" | "dashed"
    line_positions : {'left': cx|None, 'center': cx|None, 'right': cx|None}
    line_flags     : {'left': bool, 'center': bool, 'right': bool}
                      True = visible this frame, False = lost (holding prev)
    exits          : subset of ['left','center','right']  (dashed mode only)

    Dashed classification (all must hold)
    ──────────────────────────────────────
    1. >= DASH_MIN_COUNT contours
    2. Every area < DASH_MAX_AREA
    3. Horizontal span >= DASH_MIN_SPAN × width
    """

    MIN_AREA        = 50    # px² — ignore blobs smaller than this
    MAX_AREA        = 4000  # px² — ignore merged blobs (from v3)
    SIGNIFICANT_AREA= 500   # px² — real track lines are >= this (from v3)
    SAMPLE_OFFSET   = 28    # px — bilateral brightness sample distance
    BRIGHT_THR      = 120   # grayscale threshold for "bright" track surface
    MEDIAN_K        = 3     # median filter window (frames)

    # ── 3-line tracker ────────────────────────────────────────────────────────
    LINE_MAX_JUMP  = 70     # px — max allowed movement per line per frame
    LINE_MIN_SEP   = 15     # px — post-assignment order enforcement gap
    TRACK_MIN_SEP  = 40     # px — minimum gap enforced *during* triplet/pair search
    STALE_THRESH   = 15     # frames lost before clearing a line's anchor

    # ── intersection detection ────────────────────────────────────────────────
    DASH_MIN_COUNT  = 3
    DASH_MAX_AREA   = 2000  # px² — max area per contour for dashed classification
    DASH_MAX_HEIGHT = 40    # px  — contours taller than this are solid lines, not dashes
    DASH_MIN_SPAN   = 0.50  # fraction of image width

    # ── exit scanning (HSV track surface) ────────────────────────────────────
    EXIT_TRACK_RATIO = 0.15
    TRACK_HSV_LO     = np.array([78,  8, 106], np.uint8)
    TRACK_HSV_HI     = np.array([168, 76, 222], np.uint8)

    # ── lane-interior constraint (calibrated from puzzlebot_track_video.mp4) ────
    # The camera auto-white-balance shifts H dramatically frame-to-frame, so
    # H is set to the full range (0-179) — separation is by V and S only.
    # Dark gray lines: V < 75 (consistently).  Road mat: V=76-187, S=16-103.
    # White stop-line patches (if in ROI): V > 190, S < 15 — excluded by S >= 16.
    LANE_HSV_LO    = np.array([  0,  16,  76], np.uint8)
    LANE_HSV_HI    = np.array([179, 103, 187], np.uint8)
    LANE_MIN_PIX   = 800   # px of mat before trusting the centroid (mat covers ~85% of ROI)
    LANE_BLEND_THR = 40    # px deviation from lane_cx before blending toward it

    # ── persistent adaptive threshold ─────────────────────────────────────────
    # Carries T across frames so the threshold drifts gradually (±10/iter)
    # instead of re-deriving from each frame's histogram like Otsu.
    # Targets a dark-pixel % in [ADAPTIVE_DARK_MIN, ADAPTIVE_DARK_MAX].
    # Tune DARK_MIN/MAX with test_against_video.py: all 3 lines visible → check %.
    ADAPTIVE_T_INIT   = 185   # starting threshold
    ADAPTIVE_T_MIN    = 100   # floor — never let T drop below this
    ADAPTIVE_T_MAX    = 220   # ceiling — never let T rise above this
    ADAPTIVE_DARK_MIN =  2.0  # % of ROI pixels that must be dark (≈1 partial line)
    ADAPTIVE_DARK_MAX = 15.0  # % of ROI pixels that may be dark (≈3 lines + noise)

    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, debug=False):
        self.cameraWidth  = 320
        self.cameraHeight = 240
        self.debug        = debug
        self.debug_frame  = None

        # Global prev_cx (center line) used by classifier
        self.prev_cx = None
        self.history = deque(maxlen=self.MEDIAN_K)

        # Per-line state
        self.line_positions  = {'left': None, 'center': None, 'right': None}
        self.line_flags      = {'left': False, 'center': False, 'right': False}
        self._line_history   = {k: deque(maxlen=self.MEDIAN_K)
                                for k in ('left', 'center', 'right')}
        self._line_lost_frames = {'left': 0, 'center': 0, 'right': 0}

        # Intersection / exit state
        self.line_type  = "solid"
        self.exits      = []
        self._exit_mask = None

        # Brown lane-interior constraint state
        self.lane_cx         = None   # centroid x of lane interior, or None
        self.lane_correction = False  # True when cx was blended toward lane_cx

        # Persistent adaptive threshold state
        self._T_state = self.ADAPTIVE_T_INIT

    # ── main entry point ──────────────────────────────────────────────────────

    def detect_center_line(self, image, pre_cropped=False):
        """
        :param image: BGR image. If pre_cropped=False, expected 320×240 (full frame);
                      if pre_cropped=True, the image IS the bottom ROI (e.g. 320×80).
        :param pre_cropped: When True, skip internal ROI extraction — the caller already
                            cropped and resized to the relevant bottom region.
        :return: (cx, cy) of the center line.
        """
        h, w = image.shape[:2]

        if pre_cropped:
            roi     = image
            y_start = 0
        else:
            # S1: ROI — bottom third
            y_start = (2 * h) // 3
            roi     = image[y_start:h, :]

        # S2-S5: standard pipeline
        gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        binary  = self._adaptive_threshold(blurred)
        k3      = np.ones((3, 3), np.uint8)
        k5      = np.ones((5, 5), np.uint8)
        opened  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k3)
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k5)

        # S6: contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        roi_h = roi.shape[0]

        # S7: classify → track
        valid          = self._valid_contours(contours)
        self.line_type = self._classify_line(valid, w)

        if self.line_type == "dashed":
            cx = int(round(self.prev_cx)) if self.prev_cx is not None else w // 2
            cy = y_start + roi_h // 2
            self.exits = []
        else:
            cx, cy     = self._track_three_lines(valid, w, roi_h, y_start)
            self.exits = []

        # Brown lane-interior constraint: detect lane centroid and blend cx if needed
        self.lane_correction = False
        lane_cx, lane_valid = self._detect_lane_interior(roi)
        self.lane_cx = lane_cx if lane_valid else None
        if lane_valid and abs(cx - lane_cx) > self.LANE_BLEND_THR:
            cx = int(round((cx + lane_cx) / 2.0))
            self.lane_correction = True

        # Global history for classifier anchor
        self.history.append(cx)
        self.prev_cx = float(int(np.median(self.history)))

        if self.debug:
            self.debug_frame = self._build_debug(
                image, roi, gray, blurred, binary, cleaned, contours,
                cx, cy, y_start, w, roi_h
            )

        return (cx, cy)

    # ── brown lane-interior helper ────────────────────────────────────────────

    def _detect_lane_interior(self, roi):
        """
        Detect the brownish/tan lane surface in the ROI.

        Computes an HSV mask for the lane interior color and returns its
        centroid x. Used to bias the tracked center_cx toward the actual road
        when the 3-line tracker locks onto an adjacent-track line at junctions.

        Calibrate LANE_HSV_LO / LANE_HSV_HI for the specific track mat color.

        :return: (lane_cx_int, valid_bool)
        """
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LANE_HSV_LO, self.LANE_HSV_HI)
        pix  = int(mask.sum()) // 255
        if pix < self.LANE_MIN_PIX:
            return 0, False
        M = cv2.moments(mask)
        if M['m00'] == 0:
            return 0, False
        return int(round(M['m10'] / M['m00'])), True

    # ── adaptive threshold ────────────────────────────────────────────────────

    def _adaptive_threshold(self, gray_roi):
        """
        Persistent frame-to-frame threshold targeting ADAPTIVE_DARK_MIN–MAX %.

        Unlike Otsu (which re-derives from each frame's histogram), this carries
        self._T_state across frames and adjusts by ±10 per iteration.  Direction
        hysteresis stops adjustment immediately if the step direction reverses,
        preventing oscillation around the target range.

        Typical cost on a 320×80 ROI: 1–2 iterations (~0.2–0.4 ms on Nano).
        Worst-case cap: 10 iterations (~2 ms).
        """
        T         = self._T_state
        area      = gray_roi.size
        direction = 0
        for _ in range(10):
            _, binary = cv2.threshold(gray_roi, T, 255, cv2.THRESH_BINARY_INV)
            perc = 100.0 * cv2.countNonZero(binary) / area
            if perc > self.ADAPTIVE_DARK_MAX:
                # Too many dark pixels — lower T so fewer pixels qualify as dark
                if T <= self.ADAPTIVE_T_MIN or direction == 1:
                    break
                T -= 10
                direction = -1
            elif perc < self.ADAPTIVE_DARK_MIN:
                # Too few dark pixels — raise T so more pixels qualify as dark
                if T >= self.ADAPTIVE_T_MAX or direction == -1:
                    break
                T += 10
                direction = 1
            else:
                break
        self._T_state = T
        _, binary = cv2.threshold(gray_roi, T, 255, cv2.THRESH_BINARY_INV)
        return binary

    # ── contour helpers ───────────────────────────────────────────────────────

    def _valid_contours(self, contours):
        """Return [(cx_f, cy_f, area, cnt)] sorted by x, MIN_AREA ≤ area ≤ MAX_AREA."""
        valid = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_AREA or area > self.MAX_AREA:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            valid.append((M['m10'] / M['m00'], M['m01'] / M['m00'], area, cnt))
        valid.sort(key=lambda v: v[0])
        return valid

    # ── line-type classification ──────────────────────────────────────────────

    def _classify_line(self, valid, w):
        if len(valid) < self.DASH_MIN_COUNT:
            return "solid"
        areas = [v[2] for v in valid]
        if max(areas) >= self.DASH_MAX_AREA:
            return "solid"
        # Any contour as large as a real track line → solid, not intersection dashes
        if max(areas) >= self.SIGNIFICANT_AREA:
            return "solid"
        # Tall contours are solid lines — intersection dashes are small and square
        for v in valid:
            _, _, _, bh = cv2.boundingRect(v[3])
            if bh >= self.DASH_MAX_HEIGHT:
                return "solid"
        xs   = [v[0] for v in valid]
        span = max(xs) - min(xs)
        return "dashed" if span >= self.DASH_MIN_SPAN * w else "solid"

    # ── dashed-line handling ──────────────────────────────────────────────────

    def _fuse_dashes(self, valid, y_start, h):
        if not valid:
            cx = int(round(self.prev_cx)) if self.prev_cx is not None \
                 else self.cameraWidth // 2
            return cx, y_start + (h - y_start) // 2
        areas    = [v[2] for v in valid]
        total    = sum(areas) or 1.0
        fused_cx = sum(v[0] * a for v, a in zip(valid, areas)) / total
        fused_cy = sum(v[1] * a for v, a in zip(valid, areas)) / total
        return int(round(fused_cx)), y_start + int(round(fused_cy))

    def _scan_exits(self, image, y_start):
        h, w    = image.shape[:2]
        blurred = cv2.GaussianBlur(image, (5, 5), 1.4)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        tmask   = cv2.inRange(hsv, self.TRACK_HSV_LO, self.TRACK_HSV_HI)
        k3      = np.ones((3, 3), np.uint8)
        tmask   = cv2.morphologyEx(tmask, cv2.MORPH_CLOSE, k3, iterations=2)
        tmask   = cv2.morphologyEx(tmask, cv2.MORPH_OPEN,  k3, iterations=1)

        self._exit_mask    = tmask
        self._exit_y_start = y_start

        scan_mask           = tmask.copy()
        scan_mask[y_start:, :] = 0

        zone_w    = w // 3
        zone_area = y_start * zone_w

        exits = []
        for name, x0, x1 in [('left', 0, zone_w),
                               ('center', zone_w, 2*zone_w),
                               ('right', 2*zone_w, w)]:
            pix   = int(scan_mask[:, x0:x1].sum()) // 255
            ratio = pix / zone_area if zone_area > 0 else 0.0
            if ratio >= self.EXIT_TRACK_RATIO:
                exits.append(name)
        return exits

    # ── three-line tracker ────────────────────────────────────────────────────

    def _best_assignment(self, xs, w):
        """
        Assign detected x-positions (sorted) to {left, center, right} slots
        by minimising total |cx - prev_anchor| cost.

        n >= 3: try all C(n,3) triplets; assign in left→center→right order.
        n == 2: try all 3 slot-pairs (L+C, L+R, C+R).
        n == 1: assign to nearest slot.

        Returns {left: float|None, center: float|None, right: float|None}.
        """
        names = ('left', 'center', 'right')
        lp    = self.line_positions
        # When a line has no history, default to its extreme — not the image quarter.
        # This keeps uninitialized L from stealing right-side contours and vice-versa.
        defs  = {'left': 0.0, 'center': w * 0.5, 'right': float(w - 1)}
        prevs = {k: lp[k] if lp[k] is not None else defs[k] for k in names}

        n = len(xs)
        if n == 0:
            return {k: None for k in names}

        best_cost = float('inf')
        best      = {k: None for k in names}

        ms = self.TRACK_MIN_SEP

        if n >= 3:
            for i, j, k in combinations(range(n), 3):
                # Require physical separation between adjacent assigned lines
                if xs[j] - xs[i] < ms or xs[k] - xs[j] < ms:
                    continue
                cost = (abs(xs[i] - prevs['left']) +
                        abs(xs[j] - prevs['center']) +
                        abs(xs[k] - prevs['right']))
                if cost < best_cost:
                    best_cost = cost
                    best = {'left': xs[i], 'center': xs[j], 'right': xs[k]}
            # If no triplet satisfies min-sep, relax to best available
            if best_cost == float('inf'):
                for i, j, k in combinations(range(n), 3):
                    cost = (abs(xs[i] - prevs['left']) +
                            abs(xs[j] - prevs['center']) +
                            abs(xs[k] - prevs['right']))
                    if cost < best_cost:
                        best_cost = cost
                        best = {'left': xs[i], 'center': xs[j], 'right': xs[k]}
        elif n == 2:
            for sa, sb in (('left', 'center'), ('left', 'right'), ('center', 'right')):
                if xs[1] - xs[0] < ms:
                    # Two detections too close — skip slot pairs that imply adjacency
                    # (L+C and C+R require separation; L+R allows it as it's non-adjacent)
                    if sa != 'left' or sb != 'right':
                        continue
                cost = abs(xs[0] - prevs[sa]) + abs(xs[1] - prevs[sb])
                if cost < best_cost:
                    best_cost = cost
                    sc   = next(s for s in names if s not in (sa, sb))
                    best = {sa: xs[0], sb: xs[1], sc: None}
            # If still no winner (both too close), pick nearest single slot
            if best_cost == float('inf'):
                nearest = min(names, key=lambda s: abs(xs[0] - prevs[s]))
                best    = {k: None for k in names}
                best[nearest] = xs[0]
        else:
            nearest = min(names, key=lambda s: abs(xs[0] - prevs[s]))
            best    = {k: None for k in names}
            best[nearest] = xs[0]

        return best

    def _track_three_lines(self, valid, w, h, y_start):
        """
        Track all three lines via minimum-cost assignment to prev anchors.

        1. Collect detected x-positions (sorted).
        2. Assign to slots by minimum total distance to prev anchors.
        3. Per-line velocity gate + median filter.
        4. Stale anchor reset after STALE_THRESH lost frames.
        5. Enforce left < center < right ordering.
        """
        xs = sorted(v[0] for v in valid)

        detections = self._best_assignment(xs, w)

        # ── Velocity gate + median filter ─────────────────────────────────────
        for name in ('left', 'center', 'right'):
            new_cx = detections[name]
            prev   = self.line_positions[name]

            if new_cx is None:
                self.line_flags[name] = False
            elif prev is not None and abs(new_cx - prev) > self.LINE_MAX_JUMP:
                self.line_flags[name] = False
            else:
                self._line_history[name].append(new_cx)
                # Use the raw detection directly when the history is short
                # (median of 1 or 2 values) so new acquisitions take effect
                # immediately instead of being outvoted by stale history.
                if len(self._line_history[name]) < self.MEDIAN_K:
                    smoothed = int(round(new_cx))
                else:
                    smoothed = int(np.median(self._line_history[name]))
                self.line_positions[name] = float(smoothed)
                self.line_flags[name]     = True

        # ── Stale anchor reset ────────────────────────────────────────────────
        for name in ('left', 'center', 'right'):
            if self.line_flags[name]:
                self._line_lost_frames[name] = 0
            else:
                self._line_lost_frames[name] += 1
                if self._line_lost_frames[name] >= self.STALE_THRESH:
                    self.line_positions[name]    = None
                    self._line_history[name].clear()
                    self._line_lost_frames[name] = 0

        # ── Enforce ordering; reset violating pair ────────────────────────────
        lp = self.line_positions
        for a, b in (('left', 'center'), ('center', 'right')):
            if (lp[a] is not None and lp[b] is not None and
                    lp[a] >= lp[b] - self.LINE_MIN_SEP):
                for name in (a, b):
                    self.line_flags[name]        = False
                    self.line_positions[name]    = None
                    self._line_history[name].clear()
                    self._line_lost_frames[name] = 0
                break

        # ── Return center ─────────────────────────────────────────────────────
        cx_center = self.line_positions['center']
        if cx_center is None:
            lp = self.line_positions
            if lp['left'] is not None and lp['right'] is not None:
                cx_center = (lp['left'] + lp['right']) / 2.0
                # Keep the center anchor updated so next frame assigns correctly
                self.line_positions['center'] = float(cx_center)
            elif self.prev_cx is not None:
                cx_center = self.prev_cx
            else:
                cx_center = w // 2

        cy = y_start + (h - y_start) // 2
        return max(0, min(w - 1, int(round(cx_center)))), cy

    # ── bilateral brightness (used by debug; no longer used for assignment) ───

    def _is_flanked_bright(self, blurred, cx_cnt, cy_cnt):
        h, w    = blurred.shape[:2]
        y       = int(np.clip(cy_cnt, 0, h - 1))
        x_left  = int(np.clip(cx_cnt - self.SAMPLE_OFFSET, 0, w - 1))
        x_right = int(np.clip(cx_cnt + self.SAMPLE_OFFSET, 0, w - 1))

        def win_mean(x, y_):
            r0, r1 = max(0, y_ - 2), min(h, y_ + 3)
            c0, c1 = max(0, x  - 2), min(w, x  + 3)
            patch  = blurred[r0:r1, c0:c1]
            return float(patch.mean()) if patch.size > 0 else 0.0

        return win_mean(x_left, y) > self.BRIGHT_THR and \
               win_mean(x_right, y) > self.BRIGHT_THR

    # ── debug visualization ───────────────────────────────────────────────────

    def _build_debug(self, image, roi, gray, blurred, binary, cleaned,
                     contours, cx, cy, y_start, w, h):
        roi_h = roi.shape[0]

        def to_bgr(img):
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 \
                   else img.copy()

        def labeled(img, text):
            out = to_bgr(img)
            cv2.putText(out, text, (2, 11), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 0, 0), 2)
            cv2.putText(out, text, (2, 11), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (255, 255, 255), 1)
            return out

        # Tile 6: contours + all three line markers
        # Green = significant (≥SIGNIFICANT_AREA), cyan = tiny noise
        cnt_vis = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_AREA or area > self.MAX_AREA:
                continue
            col = (0, 200, 0) if area >= self.SIGNIFICANT_AREA else (0, 200, 200)
            cv2.drawContours(cnt_vis, [cnt], -1, col, 1)
        cy_roi   = cy - y_start

        if self.line_type == "dashed":
            if 0 <= cy_roi < roi_h:
                cv2.drawMarker(cnt_vis, (cx, cy_roi),
                               (0, 165, 255), cv2.MARKER_CROSS, 12, 2)
        else:
            # Draw each line marker with its own color and flag indicator
            marker_cfg = {
                'left':   ((255, 80,  80),  (0,  80, 255)),   # blue visible / dark lost
                'center': ((0,   0,  255),  (0,  0,  100)),
                'right':  ((80, 255,  80),  (0, 100,   0)),
            }
            for name, (col_vis, col_lost) in marker_cfg.items():
                pos_cx = self.line_positions[name]
                if pos_cx is None:
                    continue
                px = max(0, min(w - 1, int(round(pos_cx))))
                color = col_vis if self.line_flags[name] else col_lost
                cv2.drawMarker(cnt_vis, (px, cy_roi if 0 <= cy_roi < roi_h else roi_h // 2),
                               color, cv2.MARKER_CROSS, 10, 2)
                cv2.putText(cnt_vis, name[0].upper(), (px - 4, roi_h - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        n_sig   = sum(1 for c in contours
                      if self.MIN_AREA <= cv2.contourArea(c) <= self.MAX_AREA
                      and cv2.contourArea(c) >= self.SIGNIFICANT_AREA)
        n_valid = sum(1 for c in contours
                      if self.MIN_AREA <= cv2.contourArea(c) <= self.MAX_AREA)
        lbl_type = 'dash' if self.line_type == "dashed" else 'solid'
        lbl      = f'sig={n_sig}/{n_valid} [{lbl_type}]'
        cv2.putText(cnt_vis, lbl, (2, 11), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 0, 0), 2)
        cv2.putText(cnt_vis, lbl, (2, 11), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 255, 255), 1)

        # Tile 7: overview with exit zones (dashed) or line flags (solid)
        overview = cv2.resize(image.copy(), (roi.shape[1], roi_h))
        oh, ow   = overview.shape[:2]
        zone_w   = ow // 3

        if self.line_type == "dashed":
            sy1 = int(y_start * oh / h)
            exit_zones = [('left', 0), ('center', zone_w), ('right', 2*zone_w)]
            for name, x0 in exit_zones:
                color = (0, 255, 0) if name in self.exits else (0, 0, 100)
                cv2.rectangle(overview, (x0, 0), (x0 + zone_w - 1, sy1), color, 1)
                cv2.putText(overview, name[0].upper(),
                            (x0 + zone_w // 2 - 4, sy1 // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            exits_str = ','.join(self.exits) if self.exits else 'none'
            cv2.putText(overview, f'exits:{exits_str}', (2, oh - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 2)
            cv2.putText(overview, f'exits:{exits_str}', (2, oh - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        else:
            flags_str = ' '.join(
                f'{n[0].upper()}:{"OK" if self.line_flags[n] else "!!"}'
                for n in ('left', 'center', 'right')
            )
            cv2.putText(overview, flags_str, (2, oh - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 2)
            cv2.putText(overview, flags_str, (2, oh - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

        tiles = [
            labeled(roi,     'S1:ROI'),
            labeled(gray,    'S2:gray'),
            labeled(blurred, 'S3:blur'),
            labeled(binary,  f'S4:T={self._T_state}'),
            labeled(cleaned, 'S5:op+cl'),
            cnt_vis,
            overview,
        ]

        resized = []
        for t in tiles:
            sc = roi_h / t.shape[0]
            nw = max(1, int(t.shape[1] * sc))
            resized.append(cv2.resize(t, (nw, roi_h)))
        return np.hstack(resized)
