import time
import math
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
    # Rate-invariant velocity gate: max px/s = 70 px/frame × 30 fps.
    # At 30 fps → 70 px/frame; at 5 fps → 420 px/frame — adapts automatically.
    LINE_MAX_JUMP_PS = 2100  # px/s  (replaces fixed LINE_MAX_JUMP = 70)
    LINE_MIN_SEP   = 15     # px — post-assignment order enforcement gap
    TRACK_MIN_SEP  = 40     # px — minimum gap enforced *during* triplet/pair search
    STALE_THRESH   = 15     # frames lost before clearing a line's anchor

    # ── lane-center tracking (bad_ignment5 curve fix) ─────────────────────────
    # The robot follows the LANE CENTER (midpoint of the left/right boundary lines),
    # which stays stable through a curve, instead of the 'center' SLOT, which can
    # swap identity on a curve and teleport the target. A half-width memory lets us
    # infer the lane center when only one boundary is visible, and a rate-based jump
    # guard rejects a teleport (hold the previous lane center instead).
    LANE_CENTER_JUMP_PS    = 1500  # px/s — max plausible lane-center motion on a curve
    LANE_HALFWIDTH_MEDIAN_N = 7    # frames of (R−L)/2 to median for half-width memory

    # ── fork branch selection (pzb_not_workingcorrectly4) ─────────────────────
    # At a Y-fork the lane-center (L+R)/2 averages the two diverging branches into a
    # straight-ahead target and the robot misses the turn. Detect the fork from
    # DIVERGING HEADINGS (two branches whose heading angles splay apart, unlike
    # parallel lane boundaries which run near-parallel) and SELECT the branch that
    # best continues the approach heading (track continuity), instead of averaging.
    # All thresholds in ROI px (320×80).
    # Discriminator: a normal lane-boundary PAIR converges toward the top of the ROI
    # under perspective (top separation < base separation, ratio ≈ 0.55–0.63 measured).
    # A real FORK splays its branches OUTWARD, cancelling perspective convergence, so
    # the top stays as wide as the base (ratio ≈ 0.72–0.86 measured). Trigger only
    # when top_sep ≥ FORK_TOP_BASE_RATIO × base_sep — apparent slope-angle difference
    # ALONE does NOT work here (the oblique camera makes a normal right boundary lean
    # ~50° vs a vertical left boundary, so every lane pair would look like a fork).
    FORK_TOP_MIN_SEP    = 90    # px — branch top-x's apart by this ⇒ truly distinct lines
    FORK_TOP_BASE_RATIO = 0.70  # top_sep/base_sep ≥ this ⇒ splaying (fork), not converging pair
    FORK_MIN_BASE_SEP   = 80    # px — branches must be a real pair, not one line counted twice
    FORK_MIN_AREA       = 600   # px² — both branches must be substantial lines, not stubs
    FORK_MIN_YSPAN_FRAC = 0.60  # each branch must span ≥ this fraction of ROI height
    FORK_TIE_DEG        = 8.0   # deg — continuity tie window ⇒ fall back to least-turn
    FORK_BAND_FRAC      = 0.40  # fraction of ROI height for the lower/upper sampling bands
    APPROACH_SLOPE_EMA  = 0.4   # EMA weight for the tracked approach heading slope

    # ── intersection detection ────────────────────────────────────────────────
    # Relaxed from original (3, 2000, 40, 0.50) to handle this track's dash geometry:
    # fewer dashes visible in the bottom-80 px ROI, and some are wider than 2000 px².
    #
    # simple_track_behaviour fix: the old rule (≥2 small flat blobs spanning ≥20%
    # width with one near center) false-fired on the puzzle-mat JIGSAW SEAMS during
    # plain curves — 22% / 14% of frames mis-classified as dashed, which zeroed
    # steering mid-curve and armed the perpendicular-alignment spin → the robot drove
    # straight off the turn. A real crossing-dash row differs from scattered seams in
    # two measurable ways: (1) it has ≥3 blobs (DASH_MIN_COUNT), and (2) those blobs
    # are HORIZONTALLY CO-LINEAR — they lie within a tight y-band about their LSQ row
    # line (residual std ≤ DASH_ROW_BAND_PX). Jigsaw seams + lane-line stubs scatter
    # across the ROI height (residual std 13–22 px measured) and are now rejected.
    DASH_MIN_COUNT  = 3     # ≥3 co-linear blobs — a real dash ROW, not a stray pair
    DASH_MAX_AREA   = 3500  # px² — max area per contour for dashed classification
    DASH_MAX_HEIGHT = 35    # px  — contours taller than this are solid lines, not dashes
    DASH_MIN_SPAN   = 0.20  # fraction of image width
    DASH_CENTER_ZONE = 0.25 # at least one dash must be in the central 50% of the image
    DASH_ROW_BAND_PX = 12   # max residual std (px) of dash y about their LSQ row line.
                            # Real crossing rows measure ≤10 px; seam+stub scatter ≥13 px.

    # ── dashed alignment slope ────────────────────────────────────────────────
    # The crossing-dash slope (Δy/Δx of the dash row) measures how perpendicular
    # the robot is to the intersection. It must be fit from the DASH CANDIDATES
    # ONLY (small flat blobs) — never the continuing vertical track line — or it
    # is meaningless. Requires a clean row of dashes to be trustworthy.
    DASH_SLOPE_MIN_N    = 3     # need ≥ this many dash candidates to fit a slope
    DASH_SLOPE_MIN_SPAN = 0.25  # dash row must span ≥ this fraction of width

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
    LANE_BLEND_THR = 9999  # disabled — re-enable only after HSV re-calibration for the current track

    # ── white boundary column mask ────────────────────────────────────────────
    # Columns where >= WHITE_BOUNDARY_THRESH rows are white (V>190, S<40) are
    # treated as boundary paper and zeroed in the binary image.
    # Calibrated from bag5: boundary paper = 40-80 rows/col, mat reflections < 20.
    # bag6 showed the mat itself reaches 8-30 rows under strong lighting, which
    # caused the old threshold of 8 to false-positive and erase real line columns.
    WHITE_BOUNDARY_THRESH = 20   # rows; boundary paper: 40-80, mat noise: <20
    # The mask may only eat the outer WHITE_MASK_EDGE_FRAC of each side of the ROI.
    # Boundary paper sits at the ROI edges; a real track line near center must never
    # be erased. Without this bound, opposing left/right masks can march inward and
    # meet at center, wiping the whole line (observed in wide_angle_bag7).
    WHITE_MASK_EDGE_FRAC = 0.30  # never mask past 30% in from either edge

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

        # Lane-center tracking state (bad_ignment5 curve fix)
        self._lane_center      = None                          # last lane-center cx
        self._halfwidth_hist   = deque(maxlen=self.LANE_HALFWIDTH_MEDIAN_N)

        # Fork branch-selection state (pzb_not_workingcorrectly4)
        self._approach_slope       = 0.0    # EMA of the followed line's heading slope (Δx/Δy)
        self.branch_select_enabled = False  # node enables per-frame via fork_select_enabled param
        self.fork_active           = False  # True on a frame where a fork was selected
        self.fork_choice           = None   # 'left'/'right' branch picked (debug)
        self._fork_branches        = None   # [(cx_bot, cx_top, slope, contour), ...] (debug)

        # Intersection / exit state
        self.line_type     = "solid"
        self.exits         = []
        self._exit_mask    = None
        self.dash_slope_px   = 0.0    # Δy/Δx of crossing-dash centroids; 0 = perpendicular
        self.dash_slope_valid = False # True only when the slope was fit from a clean
                                      # row of ≥ DASH_SLOPE_MIN_N dashes spanning enough x

        # Brown lane-interior constraint state
        self.lane_cx         = None   # centroid x of lane interior, or None
        self.lane_correction = False  # True when cx was blended toward lane_cx

        # Persistent adaptive threshold state
        self._T_state = self.ADAPTIVE_T_INIT

        # Time-based velocity gate: tracks wall-clock time of last detect call
        # so the max-jump limit adapts to the actual frame interval.
        self._last_detect_t = None

    # ── tracker reset ───────────────────────────────────────────────────────

    def reset_tracker_anchors(self):
        """Clear all three-line tracker state and re-seed the center anchor.

        Called by the node on the dashed→solid transition (intersection exit) so
        the next solid-frame assignment is seeded from image center rather than
        from stale pre-intersection positions. Without this, the minimum-cost
        assignment in `_track_three_lines` can snap the center slot onto a
        crossing-dash / boundary-seam left over from the junction, producing a
        large spurious error (observed in wide_angle_bag8: cx→41, err→−119 px,
        causing a ~90° pivot on exit).
        """
        for k in ('left', 'center', 'right'):
            self.line_positions[k]    = None
            self._line_history[k].clear()
            self._line_lost_frames[k] = 0
            self.line_flags[k]        = False
        self.history.clear()
        self.prev_cx = float(self.cameraWidth // 2)
        # Reset lane-center state so a post-intersection curve starts fresh.
        self._lane_center = None
        self._halfwidth_hist.clear()
        # Reset fork/approach state — a post-intersection branch starts fresh.
        self._approach_slope = 0.0
        self.fork_active     = False
        self.fork_choice     = None
        self._fork_branches  = None

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

        roi_h = roi.shape[0]

        # White-gap mask: detect bright-white boundary paper columns and zero them out.
        # Track lines never appear behind the white boundary sheet, so any binary pixel
        # in those columns is noise from the paper surface, not a real line.
        # Threshold: WHITE_BOUNDARY_THRESH rows white in a column → boundary.
        #
        # Two safeguards prevent the mask from erasing a real line (wide_angle_bag7):
        #   1. Edge bound — each side may only eat the outer WHITE_MASK_EDGE_FRAC of the
        #      ROI, so opposing masks can never meet and wipe a centered line.
        #   2. Survivor guard — the masked binary is only committed if at least one
        #      valid track contour survives it; otherwise the unmasked binary is kept.
        #      A real dark line is always preferable to a full stop.
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        _white_mask = cv2.inRange(roi_hsv,
                                  np.array([0, 0, 190], np.uint8),
                                  np.array([179, 40, 255], np.uint8))
        _white_col_counts = _white_mask.sum(axis=0) // 255
        _boundary_cols = np.where(_white_col_counts >= self.WHITE_BOUNDARY_THRESH)[0]
        if len(_boundary_cols) > 0:
            _w     = roi.shape[1]
            _mid   = _w // 2
            _l_lim = int(_w * self.WHITE_MASK_EDGE_FRAC)            # left mask stops here
            _r_lim = int(_w * (1.0 - self.WHITE_MASK_EDGE_FRAC))   # right mask starts here
            _right = _boundary_cols[_boundary_cols >= _mid]
            _left  = _boundary_cols[_boundary_cols < _mid]

            _masked = binary.copy()
            if len(_right) > 0:
                _r0 = max(_right[0], _r_lim)
                _masked[:, _r0:] = 0
            if len(_left) > 0:
                _l1 = min(_left[-1] + 1, _l_lim)
                _masked[:, :_l1] = 0

            # Survivor guard: commit the mask only if a real line remains.
            if self._count_track_contours(_masked, roi_h) > 0:
                binary = _masked

        k3      = np.ones((3, 3), np.uint8)
        k5      = np.ones((5, 5), np.uint8)
        opened  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k3)
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k5)

        # S6: contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # S7: classify → track
        # Far-field filter: reject contours in the top 15% of the ROI — these are
        # distant objects (blue boundary tape, far track features) that are never
        # near-field lines the robot should react to.
        valid = [v for v in self._valid_contours(contours) if v[1] >= roi_h * 0.15]

        self.line_type = self._classify_line(valid, w)

        # Compute time-based velocity gate (Team2 pattern): pixels/s × dt_seconds.
        # This makes the gate FPS-invariant: 2100 px/s ÷ 30 fps = 70 px/frame,
        # 2100 px/s ÷ 5.7 fps = 368 px/frame — valid turns are no longer rejected.
        _now = time.monotonic()
        _dt  = (_now - self._last_detect_t) if self._last_detect_t is not None else 1.0 / 30.0
        self._last_detect_t = _now
        _max_jump = self.LINE_MAX_JUMP_PS * _dt

        if self.line_type == "dashed":
            cx, cy = self._fuse_dashes(valid, y_start, roi_h)
            self.exits = []
        else:
            cx, cy     = self._track_three_lines(valid, w, roi_h, y_start, _max_jump)
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
                T = max(self.ADAPTIVE_T_MIN, T - 10)
                direction = -1
            elif perc < self.ADAPTIVE_DARK_MIN:
                # Too few dark pixels — raise T so more pixels qualify as dark
                if T >= self.ADAPTIVE_T_MAX or direction == -1:
                    break
                T = min(self.ADAPTIVE_T_MAX, T + 10)
                direction = 1
            else:
                break
        self._T_state = T
        _, binary = cv2.threshold(gray_roi, T, 255, cv2.THRESH_BINARY_INV)
        return binary

    # ── contour helpers ───────────────────────────────────────────────────────

    def _count_track_contours(self, binary, roi_h):
        """Count valid near-field track contours a binary would yield.

        Runs the same morphology → contour → size → far-field filter the main
        pipeline uses, so the white-mask survivor guard judges a mask candidate
        by exactly what the detector would later see. Returns an int count.
        """
        k3      = np.ones((3, 3), np.uint8)
        k5      = np.ones((5, 5), np.uint8)
        opened  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k3)
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k5)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        return sum(1 for v in self._valid_contours(contours)
                   if v[1] >= roi_h * 0.15)

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

    def _dash_candidates(self, valid):
        """Return the small, flat blobs that look like crossing dashes.

        A dash candidate is NOT tall (bh < DASH_MAX_HEIGHT) and NOT a vertical
        stub (bh < bw). Tall+large contours are real solid lines, not dashes.
        Returns (candidates, has_solid_segment) — the second flag is True if a
        contour proves a solid line is present (tall AND large), which vetoes
        dashed classification. Shared by `_classify_line` and `_fuse_dashes` so
        the alignment slope is fit from EXACTLY the same dash set.
        """
        candidates = []
        has_solid = False
        for v in valid:
            _, _, bw, bh = cv2.boundingRect(v[3])
            area = v[2]
            is_tall  = bh >= self.DASH_MAX_HEIGHT
            is_stub  = bw > 0 and bh >= bw      # not flat → lane marker / vertical stub
            is_large = area >= self.DASH_MAX_AREA
            if is_tall and is_large:
                has_solid = True
            if not is_tall and not is_stub:
                candidates.append(v)
        return candidates, has_solid

    def _classify_line(self, valid, w):
        if len(valid) < self.DASH_MIN_COUNT:
            return "solid"

        # Separate contours into dash candidates (small flat blobs) and a solid
        # veto (tall+large contour = a real solid line). With an oblique forward
        # camera, solid lines appear as tall narrow stubs mixed in with dashes.
        dash_candidates, has_solid = self._dash_candidates(valid)
        if has_solid:
            return "solid"

        if len(dash_candidates) < self.DASH_MIN_COUNT:
            return "solid"

        xs   = [v[0] for v in dash_candidates]
        span = max(xs) - min(xs)
        if span < self.DASH_MIN_SPAN * w:
            return "solid"

        # Reject corner markers visible only at the extreme image edges.
        # Genuine crossing dashes always have at least one blob in the central zone.
        center_lo = w * self.DASH_CENTER_ZONE
        center_hi = w * (1.0 - self.DASH_CENTER_ZONE)
        if not any(center_lo <= v[0] <= center_hi for v in dash_candidates):
            return "solid"

        # Horizontal co-linearity gate (simple_track_behaviour fix): a genuine
        # crossing is a ROW of dashes lying on one near-horizontal line, so the
        # candidate y's cluster tightly about their least-squares row line. The
        # puzzle-mat jigsaw seams that previously false-triggered this scatter
        # across the ROI height instead (residual std 13–22 px vs ≤10 px for a real
        # row). Reject when the residual std exceeds DASH_ROW_BAND_PX.
        xs_a = np.asarray(xs, dtype=np.float64)
        ys_a = np.asarray([v[1] for v in dash_candidates], dtype=np.float64)
        row_slope, row_b = np.polyfit(xs_a, ys_a, 1)
        row_resid_std = float((ys_a - (row_slope * xs_a + row_b)).std())
        if row_resid_std > self.DASH_ROW_BAND_PX:
            return "solid"

        return "dashed"

    # ── dashed-line handling ──────────────────────────────────────────────────

    def _fuse_dashes(self, valid, y_start, h):
        if not valid:
            self.dash_slope_px = 0.0
            self.dash_slope_valid = False
            cx = int(round(self.prev_cx)) if self.prev_cx is not None \
                 else self.cameraWidth // 2
            return cx, y_start + (h - y_start) // 2
        areas    = [v[2] for v in valid]
        total    = sum(areas) or 1.0
        fused_cx = sum(v[0] * a for v, a in zip(valid, areas)) / total
        fused_cy = sum(v[1] * a for v, a in zip(valid, areas)) / total

        # Crossing-dash slope (Δy/Δx of the dash row): 0 = robot perpendicular to
        # the dashes. CRITICAL: fit ONLY the dash candidates (small flat blobs),
        # NEVER the continuing vertical track line — mixing them makes the slope
        # meaningless (observed in bad_alginment2: −25°↔+10° swings frame to frame).
        # Use a least-squares line over the dash centroids, and only mark it valid
        # when there is a clean row (≥ DASH_SLOPE_MIN_N dashes spanning enough x).
        dash_cands, _ = self._dash_candidates(valid)
        self.dash_slope_px   = 0.0
        self.dash_slope_valid = False
        if len(dash_cands) >= self.DASH_SLOPE_MIN_N:
            xs = np.array([v[0] for v in dash_cands], dtype=np.float64)
            ys = np.array([v[1] for v in dash_cands], dtype=np.float64)
            x_span = float(xs.max() - xs.min())
            if x_span >= self.DASH_SLOPE_MIN_SPAN * self.cameraWidth:
                # slope = dy/dx via least squares (robust to >2 dashes + 1 outlier)
                slope = float(np.polyfit(xs, ys, 1)[0])
                self.dash_slope_px    = slope
                self.dash_slope_valid = True

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

    # ── fork branch selection (pzb_not_workingcorrectly4) ─────────────────────

    def _branch_descriptor(self, cnt, h):
        """For one contour, return (cx_bottom, cx_top, slope_px, yspan) in ROI px.

        cx_bottom / cx_top = mean x of the contour points in the bottom / top band
        (FORK_BAND_FRAC of ROI height). slope_px = Δx/Δy heading (x as a function of
        y) fit over ALL contour points via least squares; sign = turn direction
        (positive ⇒ line leans toward larger x as it goes UP the image). yspan = the
        contour's vertical extent (used to reject short stubs as fork branches).
        Returns None if the contour does not span both bands.
        """
        pts = cnt.reshape(-1, 2).astype(np.float64)
        xs, ys = pts[:, 0], pts[:, 1]
        band = self.FORK_BAND_FRAC * h
        y_lo_max = h - band          # bottom band: y >= this (near the robot)
        y_hi_min = band              # top band:    y <= this (far from the robot)
        bot = pts[ys >= y_lo_max]
        top = pts[ys <= y_hi_min]
        if len(bot) == 0 or len(top) == 0:
            return None
        cx_bottom = float(bot[:, 0].mean())
        cx_top    = float(top[:, 0].mean())
        yspan = float(ys.max() - ys.min())
        # slope = dx/dy (lines are near-vertical, so fit x on y). Guard tiny y-span.
        if yspan < 1.0:
            return None
        slope_px = float(np.polyfit(ys, xs, 1)[0])
        return (cx_bottom, cx_top, slope_px, yspan)

    @staticmethod
    def _slope_to_deg(slope_px):
        # ROI is anisotropically resized; the dashed-alignment path divides by 0.889
        # to recover a real angle. Here we only COMPARE branch slopes to each other
        # and to the approach slope (all in the same ROI-px space), so a plain atan
        # is a monotonic, sufficient angle proxy for the thresholds.
        return math.degrees(math.atan(slope_px))

    def _detect_fork(self, valid, w, h):
        """Splaying-pair fork detector.

        Discriminator (see FORK_TOP_BASE_RATIO): a normal lane-boundary PAIR converges
        toward the top of the ROI under perspective (top_sep < base_sep), whereas a
        FORK splays its branches outward so the top stays ~as wide as the base
        (top_sep ≈ base_sep, even slightly wider). Apparent slope-angle difference is
        NOT usable on this oblique camera — a vertical left boundary vs a perspective-
        leaned right boundary already differ ~50° in a plain lane.

        A fork requires two SUBSTANTIAL branches (area ≥ FORK_MIN_AREA, span ≥
        FORK_MIN_YSPAN_FRAC of ROI height, distinct: base_sep ≥ FORK_MIN_BASE_SEP and
        top_sep ≥ FORK_TOP_MIN_SEP) whose top/base separation ratio ≥
        FORK_TOP_BASE_RATIO (they do not converge with perspective).

        Returns (is_fork, [branchA, branchB]) — the widest-at-top qualifying pair.
        Each branch is (cx_bottom, cx_top, slope_px, yspan). When not: (False, []).
        """
        min_yspan = self.FORK_MIN_YSPAN_FRAC * h
        descs = []
        for v in valid:
            if v[2] < self.FORK_MIN_AREA:
                continue
            d = self._branch_descriptor(v[3], h)
            if d is not None and d[3] >= min_yspan:
                descs.append(d)
        if len(descs) < 2:
            return False, []

        best = None  # (top_sep, A, B)
        for i in range(len(descs)):
            for j in range(i + 1, len(descs)):
                A, B = descs[i], descs[j]
                base_sep = abs(A[0] - B[0])
                top_sep  = abs(A[1] - B[1])
                if (base_sep >= self.FORK_MIN_BASE_SEP and
                        top_sep >= self.FORK_TOP_MIN_SEP and
                        top_sep >= self.FORK_TOP_BASE_RATIO * base_sep):
                    if best is None or top_sep > best[0]:
                        best = (top_sep, A, B)
        if best is None:
            return False, []
        return True, [best[1], best[2]]

    def _select_fork_branch(self, branches, ref_cx):
        """Pick the branch that best continues the line the robot is ALREADY on.

        Continuity is dominated by BASE PROXIMITY: the branch whose base (cx_bottom)
        is nearest the reference center the robot was tracking (ref_cx = the prior
        lane-center / prev_cx) is the branch the robot is physically on. This is far
        more stable than matching the branch heading slope to a lagging approach-EMA
        — once the robot is on the left curve, the left branch's slope steepens past
        the EMA and a slope-only match wrongly flips to the straighter (wrong) branch
        (observed in pzb_not_workingcorrectly4 at t≈22.65). The approach-heading slope
        is used only as a TIEBREAK when both bases are near-equidistant.

        Returns (cx_bottom, slope_px, choice_str) for the chosen branch.
        """
        appr = self._approach_slope
        by_base = sorted(branches, key=lambda b: abs(b[0] - ref_cx))
        best, second = by_base[0], by_base[1]
        # Near-equidistant bases → tiebreak by heading continuity (slope vs approach),
        # then by least-turn (straightest) if still tied.
        if abs(abs(best[0] - ref_cx) - abs(second[0] - ref_cx)) <= self.FORK_TOP_MIN_SEP:
            by_head = sorted(branches, key=lambda b: abs(b[2] - appr))
            h0, h1 = by_head[0], by_head[1]
            if abs(abs(h0[2] - appr) - abs(h1[2] - appr)) <= \
                    abs(math.tan(math.radians(self.FORK_TIE_DEG))):
                best = min(branches, key=lambda b: abs(b[2]))
            else:
                best = h0
        # Label by which side the chosen branch leans (smaller cx_bottom ⇒ left).
        other = max(branches, key=lambda b: abs(b[0] - best[0]))
        choice = 'left' if best[0] <= other[0] else 'right'
        return best[0], best[2], choice

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

    def _track_three_lines(self, valid, w, h, y_start, max_jump):
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
            elif prev is not None and abs(new_cx - prev) > max_jump:
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

        # ── Enforce ordering; flag violating pair lost but keep anchors ──────
        # Positions and history are preserved so the next frame's assignment
        # still has a realistic reference. Clearing them forces fallback to
        # extreme defaults (0 / 160 / 319) causing wrong-slot assignments.
        lp = self.line_positions
        for a, b in (('left', 'center'), ('center', 'right')):
            if (lp[a] is not None and lp[b] is not None and
                    lp[a] >= lp[b] - self.LINE_MIN_SEP):
                for name in (a, b):
                    self.line_flags[name] = False
                    # _line_lost_frames increments in the stale block above;
                    # stale-reset will expire the anchor after STALE_THRESH frames.
                break

        # ── Return lane-center (bad_ignment5 curve fix) ───────────────────────
        # The robot follows the LANE CENTER (midpoint of the boundary lines), which
        # stays stable through a curve, NOT the 'center' SLOT, which can swap
        # identity on a curve and teleport the target (observed 117→244 on a bend).
        lp = self.line_positions
        L, C, R = lp['left'], lp['center'], lp['right']
        L_ok = self.line_flags['left']   and L is not None
        R_ok = self.line_flags['right']  and R is not None

        # ── Fork branch selection (pzb_not_workingcorrectly4) ─────────────────
        # At a Y-fork, (L+R)/2 averages the two diverging branches into a
        # straight-ahead target and the robot misses the turn. Detect the fork from
        # diverging-branch geometry and steer toward the branch that CONTINUES the
        # approach heading. Gated off while the node is squaring up to a dashed
        # crossing (branch_select_enabled=False) so it never fights the alignment.
        self.fork_active    = False
        self.fork_choice    = None
        self._fork_branches = None
        fork_cx = None
        if self.branch_select_enabled:
            is_fork, branches = self._detect_fork(valid, w, h)
            if is_fork:
                # Reference = the line the robot is already on (prior lane-center).
                ref_cx = self._lane_center if self._lane_center is not None else (
                    self.prev_cx if self.prev_cx is not None else w / 2.0)
                fork_cx, _bslope, choice = self._select_fork_branch(branches, ref_cx)
                self.fork_active    = True
                self.fork_choice    = choice
                self._fork_branches = branches

        if fork_cx is not None:
            # Steer toward the chosen branch's base; do NOT learn half-width here
            # (the branches are diverging, not a parallel lane pair).
            cx_center = float(fork_cx)
        elif L_ok and R_ok:
            # Both boundaries: lane-center = midpoint; learn the lane half-width.
            cx_center = (L + R) / 2.0
            self._halfwidth_hist.append((R - L) / 2.0)
        elif (L_ok or R_ok) and len(self._halfwidth_hist) > 0:
            # One boundary: infer the lane-center from the learned half-width so we
            # don't snap to a slot when the other boundary drops out on a curve.
            hw = float(np.median(self._halfwidth_hist))
            cx_center = (L + hw) if L_ok else (R - hw)
        elif C is not None and self.line_flags['center']:
            # No usable boundary pair — fall back to the center slot if it is real.
            cx_center = float(C)
        elif self._lane_center is not None:
            cx_center = self._lane_center
        elif self.prev_cx is not None:
            cx_center = self.prev_cx
        else:
            cx_center = w / 2.0

        # Continuity / jump guard: a slot-swap can still teleport the target. Reject
        # a one-frame lane-center move larger than a curve-plausible bound and hold
        # the previous lane-center instead. Bound scales with dt like the line gate
        # (max_jump is LINE_MAX_JUMP_PS×dt px/frame; scale to the lane bound).
        lane_bound = max_jump * (self.LANE_CENTER_JUMP_PS / self.LINE_MAX_JUMP_PS)
        if (self._lane_center is not None
                and abs(cx_center - self._lane_center) > lane_bound):
            # Move at most lane_bound toward the new estimate (damp the teleport).
            step = lane_bound if cx_center > self._lane_center else -lane_bound
            cx_center = self._lane_center + step

        self._lane_center = float(cx_center)
        # Keep the center slot anchored at the lane-center so next-frame assignment
        # has a realistic reference (prevents the swap from re-seeding badly).
        self.line_positions['center'] = float(cx_center)

        # Update the approach-heading EMA: slope of the contour the robot is now
        # following (the one whose base is closest to the chosen lane-center). This
        # is the "where I'm heading" estimate the fork selector matches branches to.
        self._update_approach_slope(valid, cx_center, h)

        cy = y_start + (h - y_start) // 2
        return max(0, min(w - 1, int(round(cx_center)))), cy

    def _update_approach_slope(self, valid, cx_center, h):
        """EMA-track the heading slope (Δx/Δy) of the line nearest cx_center."""
        best_d = None
        best_slope = None
        for v in valid:
            if v[2] < self.FORK_MIN_AREA:
                continue
            d = self._branch_descriptor(v[3], h)
            if d is None:
                continue
            dist = abs(d[0] - cx_center)
            if best_d is None or dist < best_d:
                best_d = dist
                best_slope = d[2]
        if best_slope is not None:
            a = self.APPROACH_SLOPE_EMA
            self._approach_slope = (1.0 - a) * self._approach_slope + a * best_slope

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

            # Fork overlay (pzb_not_workingcorrectly4): draw the two diverging branch
            # headings and highlight the CHOSEN one in magenta, plus a fork:L/R label.
            if self.fork_active and self._fork_branches:
                # Mark the chosen branch base (already decided this frame; recompute
                # against the committed lane-center for the overlay).
                ref_cx = self._lane_center if self._lane_center is not None else (
                    self.prev_cx if self.prev_cx is not None else w / 2.0)
                cb, cs, _choice = self._select_fork_branch(self._fork_branches, ref_cx)
                chosen_bot = cb
                for (cx_bot, cx_top, _slope) in self._fork_branches:
                    p0 = (max(0, min(w - 1, int(round(cx_bot)))), roi_h - 1)
                    p1 = (max(0, min(w - 1, int(round(cx_top)))), 0)
                    is_chosen = abs(cx_bot - chosen_bot) < 1e-3
                    col = (255, 0, 255) if is_chosen else (180, 180, 180)
                    cv2.line(cnt_vis, p0, p1, col, 2 if is_chosen else 1)
                cv2.putText(cnt_vis, f'fork:{self.fork_choice}', (2, roi_h - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 2)
                cv2.putText(cnt_vis, f'fork:{self.fork_choice}', (2, roi_h - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)

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
