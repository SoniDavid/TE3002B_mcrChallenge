import time
import math
from collections import deque
from itertools import combinations
import numpy as np
import cv2


# ── miniretoS8 reference center-pick helpers (ROUND 8) ────────────────────────
# Faithful ports of line_detector2.py (_threshold_dark / _scan_band_centers /
# _fallback_contour_center). The zebra guard is intentionally omitted.
def _odd_ge(v):
    v = max(3, int(v))
    return v if v % 2 == 1 else v + 1


def _ref_threshold_dark(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    block = _odd_ge(min(151, max(41, roi.shape[1] // 5)))
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, block, 8)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    dark_v = cv2.inRange(hsv[:, :, 2], 0, 115)
    mask = cv2.bitwise_or(adaptive, dark_v)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    return mask


def _ref_find_runs(b):
    runs, in_run, s = [], False, 0
    for i, v in enumerate(b):
        if v and not in_run:
            s, in_run = i, True
        elif not v and in_run:
            runs.append((s, i - 1)); in_run = False
    if in_run:
        runs.append((s, len(b) - 1))
    return runs


def _ref_scan_bands(mask, prev_cx_roi):
    roi_h, roi_w = mask.shape[:2]
    centers = []
    expected = prev_cx_roi if prev_cx_roi is not None else roi_w / 2.0
    for r0, r1, bw in [(0.82, 1.00, 1.00), (0.68, 0.84, 0.85), (0.54, 0.70, 0.65),
                       (0.40, 0.56, 0.45), (0.26, 0.42, 0.30)]:
        y0, y1 = int(r0 * roi_h), int(r1 * roi_h)
        band = mask[y0:y1, :]
        if band.size == 0:
            continue
        if cv2.countNonZero(band) / float(band.size) > 0.55:
            continue
        col = np.sum(band > 0, axis=0).astype(np.float32)
        if float(np.max(col)) < 3.0:
            continue
        col = cv2.GaussianBlur(col.reshape(1, -1), (1, 31), 0).flatten()
        thr = max(3.0, 0.32 * float(np.max(col)))
        valid = []
        for x0, x1 in _ref_find_runs(col > thr):
            width = x1 - x0 + 1
            if width < 5 or width > 0.58 * roi_w:
                continue
            valid.append((0.5 * (x0 + x1), float(np.max(col[x0:x1 + 1])), width))
        if not valid:
            continue
        valid.sort(key=lambda r: r[0])
        if len(valid) >= 3:
            cx, peak, width = valid[len(valid) // 2]
        else:
            cx, peak, width = min(valid, key=lambda r: abs(r[0] - expected))
        expected = cx
        wq = 1.0 - min(1.0, width / (0.58 * roi_w))
        centers.append((cx, bw * peak * (1.0 + 0.1 * wq)))
    return centers


def _ref_fallback_center(mask, prev_cx_roi):
    roi_h, roi_w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    expected = prev_cx_roi if prev_cx_roi is not None else roi_w / 2.0
    best, best_score = None, -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 120:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 0.65 * roi_w and h < 0.30 * roi_h:
            continue
        if h < 6:
            continue
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = float(M['m10'] / M['m00'])
        score = area * (0.55 * (y + h) / roi_h + 0.45 * (1.0 - min(1.0, abs(cx - expected) / (0.5 * roi_w))))
        if score > best_score:
            best_score, best = score, cx
    return best


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

    # ── jigsaw puzzle-seam rejection (opt-bags center-loss fix) ────────────────
    # The white puzzle-piece cutout seams on the mat appear as thin, wiggly, closed
    # loops in the binary and were being assigned to L/C/R line slots — on a sharp
    # curve they competed with the real boundary and made the 'center' flicker
    # (opt9 cx 113↔160). A real track line / crossing dash is an elongated, well-
    # FILLED stripe (high extent AND high solidity); a seam squiggle doubles back so
    # BOTH are low. Reject only SMALL contours (real boundaries are larger and never
    # touched) and only when BOTH metrics are low — conservative, so real curved line
    # fragments and solid dashes survive. Tuned from opt9 t40-41 seam contours.
    SEAM_MAX_AREA     = 500   # px² — only test sub-significant blobs
    SEAM_MAX_EXTENT   = 0.32  # area/(bbox area) below this = thin & wiggly, not a stripe
    SEAM_MAX_SOLIDITY = 0.45  # area/(hull area) below this = doubles back, not a line
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

    # ── lane-center tracking (lean rewrite) ───────────────────────────────────
    # The robot follows the LANE CENTER = midpoint of the left/right boundaries.
    # half-width memory infers where center is when only one boundary is visible.
    LANE_HALFWIDTH_MEDIAN_N = 7    # frames of (R−L)/2 to median for half-width memory
    # When only ONE boundary is visible, move the center target only this fraction of
    # the way from the prior center toward the inferred (one-line+half-width) target.
    # <1 keeps the robot near center and pulls the lane back into view instead of
    # committing to a far single-line offset and drifting off (no_turning_well fix).
    SINGLE_LINE_PULL = 0.5

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

        # Lane-center tracking state (lean rewrite)
        self.prev_center     = None                            # last lane-center cx
        self._halfwidth_hist = deque(maxlen=self.LANE_HALFWIDTH_MEDIAN_N)
        # True when this frame's center came from BOTH boundaries (a confident lane
        # center). False ⇒ <2 boundaries → the node should slow to regain the lane.
        self.centering       = False
        # Retained no-op for node compatibility (fork selection removed in lean rewrite).
        self.branch_select_enabled = False

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
        self.prev_center = None
        self._halfwidth_hist.clear()
        self.centering   = False

    # ── miniretoS8 reference center-pick (ROUND 8) ────────────────────────────
    def ref_center_line(self, roi_bgr, prev_direction):
        """Reference 5-band center-pick → normalized direction ∈ [-1,1] (negative=left).

        Faithful port of line_detector2.py (_threshold_dark + _scan_band_centers +
        _fallback_contour_center); the zebra guard is OMITTED (the dashed-FSM + YOLO
        handle crossings). Independent of the slot tracker, used as the LINE-FOLLOW
        steering source when the node runs in reference-control mode. Returns
        (direction, found).
        """
        h, w = roi_bgr.shape[:2]
        crop_x0, crop_x1 = int(w * 0.05), int(w * 0.95)
        roi = roi_bgr[:, crop_x0:crop_x1]
        mask = _ref_threshold_dark(roi)
        roi_w = mask.shape[1]

        prev_cx_roi = None
        if prev_direction is not None:
            prev_cx_full = (prev_direction + 1.0) * (w / 2.0)
            prev_cx_roi = max(0.0, min(float(roi_w - 1), prev_cx_full - crop_x0))

        centers = _ref_scan_bands(mask, prev_cx_roi)
        cx_roi = roi_w / 2.0
        found = False
        if centers:
            ws = sum(c * s for c, s in centers)
            wt = sum(s for _, s in centers)
            cx_roi = ws / max(1e-6, wt)
            found = (min(1.0, len(centers) / 4.0) > 0.15) or len(centers) >= 1
        else:
            fb = _ref_fallback_center(mask, prev_cx_roi)
            if fb is not None:
                cx_roi = fb
                found = True

        cx_full = max(0.0, min(float(w - 1), cx_roi + crop_x0))
        direction = (cx_full / (w / 2.0)) - 1.0
        return max(-1.0, min(1.0, float(direction))), found

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

        # Brown lane-interior constraint: detect lane centroid and blend cx if needed.
        # SKIPPED entirely when the blend is disabled (LANE_BLEND_THR == 9999): the
        # detector ran an HSV cvtColor + inRange + moments every frame and then threw the
        # result away (CPU-fix). Reuses the ROI-HSV already computed for the white mask.
        self.lane_correction = False
        self.lane_cx = None
        if self.LANE_BLEND_THR < 9999:
            lane_cx, lane_valid = self._detect_lane_interior(roi, roi_hsv)
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

    def _detect_lane_interior(self, roi, roi_hsv=None):
        """
        Detect the brownish/tan lane surface in the ROI.

        Computes an HSV mask for the lane interior color and returns its
        centroid x. Used to bias the tracked center_cx toward the actual road
        when the 3-line tracker locks onto an adjacent-track line at junctions.

        Calibrate LANE_HSV_LO / LANE_HSV_HI for the specific track mat color.

        :param roi_hsv: optional precomputed BGR2HSV of `roi` (the caller already
                        converts it for the white mask) — reused to avoid a second
                        cvtColor per frame.
        :return: (lane_cx_int, valid_bool)
        """
        hsv  = roi_hsv if roi_hsv is not None else cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
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
            # Jigsaw puzzle-seam rejection (opt-bags center-loss fix): drop small, thin,
            # wiggly closed loops that are mat seams, not track lines. Only small blobs
            # are tested (real boundaries are larger and pass straight through), and BOTH
            # fill-extent AND solidity must be low, so elongated line fragments and solid
            # crossing dashes (both high-extent) are never rejected.
            if area < self.SEAM_MAX_AREA:
                _x, _y, bw, bh = cv2.boundingRect(cnt)
                extent   = area / float(bw * bh) if bw * bh > 0 else 1.0
                hull_a   = cv2.contourArea(cv2.convexHull(cnt))
                solidity = area / hull_a if hull_a > 0 else 1.0
                if extent < self.SEAM_MAX_EXTENT and solidity < self.SEAM_MAX_SOLIDITY:
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

        # ── Velocity gate (CLAMP, don't discard) + median filter ──────────────
        # Lean-rewrite (no_turning fix): the old gate DROPPED a line whenever its
        # detection jumped > max_jump, and the ordering block flagged whole pairs lost.
        # On bag2 that discarded detectable lines in 18-23% of frames (≥2 contours
        # present but tracker reported ≤1), so the robot rode a single boundary and
        # drifted off. Instead of discarding, CLAMP the accepted position to at most
        # max_jump from the previous anchor: a real fast move still advances (capped)
        # and the line stays TRACKED, while a one-frame teleport is bounded. The line
        # is only "not visible" when there is genuinely no detection for that slot.
        for name in ('left', 'center', 'right'):
            new_cx = detections[name]
            prev   = self.line_positions[name]

            if new_cx is None:
                self.line_flags[name] = False
                continue
            if prev is not None and abs(new_cx - prev) > max_jump:
                new_cx = prev + (max_jump if new_cx > prev else -max_jump)
            self._line_history[name].append(new_cx)
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

        # ── Enforce ordering; drop only the LESS-trusted side of a bad pair ────
        # Lean-rewrite: the old code flagged BOTH lines of an out-of-order pair lost,
        # throwing away a good boundary on every transient overlap. Keep the line with
        # the longer history (more trusted) visible and drop only the other, so we never
        # lose both boundaries to a single ordering glitch. Anchors are preserved.
        lp = self.line_positions
        for a, b in (('left', 'center'), ('center', 'right')):
            if (lp[a] is not None and lp[b] is not None and
                    lp[a] >= lp[b] - self.LINE_MIN_SEP):
                drop = a if len(self._line_history[a]) <= len(self._line_history[b]) else b
                self.line_flags[drop] = False
                break

        # ── Lane center = midpoint of the two boundaries ──────────────────────
        # LEAN REWRITE (no_turning_well fix). The robot follows the LANE CENTER, the
        # midpoint of the left/right boundaries. The previous version stacked a learned
        # half-width, a continuity-sign single-boundary inference, a fork branch
        # selector, a dt-scaled jump guard and a lane-center median — ~100 lines that
        # (a) cost memory/CPU and (b) made the robot ride one boundary off-center when
        # the inference picked the wrong side. Replaced with three plain cases:
        #   both boundaries  → (L+R)/2 and learn the half-width,
        #   one boundary     → BIAS BACK TOWARD CENTER (do not commit to a far offset):
        #                      step the target from the prior center toward the lane side
        #                      by a bounded amount so the missing boundary comes back into
        #                      frame, instead of locking onto a single line and drifting,
        #   none             → hold the prior center.
        # cx_pull (set here) tells the node to ALSO slow down while <2 boundaries are
        # visible so the lane is regained rather than driven past.
        lp = self.line_positions
        L, C, R = lp['left'], lp['center'], lp['right']
        L_ok = self.line_flags['left']  and L is not None
        R_ok = self.line_flags['right'] and R is not None

        prev_center = (self.prev_center if self.prev_center is not None
                       else (self.prev_cx if self.prev_cx is not None else w / 2.0))

        if L_ok and R_ok:
            cx_center = (L + R) / 2.0
            self._halfwidth_hist.append((R - L) / 2.0)
            self.centering = True
        elif (L_ok or R_ok):
            # One boundary: aim for where the center SHOULD be (visible line offset by the
            # learned half-width toward the lane), but only move PART WAY from the prior
            # center so a wrong/short-lived single-line reading can't throw the target to
            # the far edge. This keeps us near center and pulls the lane back into view.
            X = float(L if L_ok else R)
            hw = (float(np.median(self._halfwidth_hist))
                  if len(self._halfwidth_hist) > 0 else w * 0.4)
            side = 1.0 if X < prev_center else -1.0     # center is on the lane side of X
            target = X + side * hw
            cx_center = prev_center + self.SINGLE_LINE_PULL * (target - prev_center)
            self.centering = False
        elif C is not None and self.line_flags['center']:
            cx_center = float(C)
            self.centering = False
        else:
            cx_center = prev_center
            self.centering = False

        # Light smoothing: one-frame median against the previous center kills a single
        # slot-swap frame without the multi-frame lag the old median stack added.
        if self.prev_center is not None:
            cx_center = 0.5 * (cx_center + self.prev_center) if abs(cx_center - self.prev_center) > w * 0.25 else cx_center

        self.prev_center = float(cx_center)
        self.line_positions['center'] = float(cx_center)

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
