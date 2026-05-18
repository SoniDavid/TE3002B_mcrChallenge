#!/usr/bin/env python3
"""
Linear MPC node for Image-Based Visual Servoing (IBVS).

State:    x = [eu, ev, ea, v, omega]   (pixel errors + scale error + actual body velocities)
Control:  u = [v_cmd, omega_cmd]       (linear and angular velocity commands)

Prediction model — augmented state with first-order velocity dynamics:

    x_{k+1} = A(L) * x_k + B * u_k

    A is 5×5: image rows couple to velocity states via interaction matrix L;
              velocity rows model closed-loop first-order response.
    B is 5×2: only velocity states are directly driven by commands.
              Image states are driven through the velocity states (captures actuator lag).

Linearisation of L is refreshed each cycle from current (eu, ev) and depth
estimated from the measured area — LTV-MPC with depth adaptation.
The QP is condensed over the horizon N and solved with OSQP, which is kept
alive between cycles so warm-starting actually works.

Subscribes:
  /visual_features  (std_msgs/Float64MultiArray)  [eu, ev, ea, confidence]
  /robot_vel        (geometry_msgs/TwistStamped)  actual body velocity from MCU

Publishes:
  /cmd_vel_desired  (geometry_msgs/Twist)
"""

import signal
import time
import numpy as np
import scipy.sparse as sp

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import Float64MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

try:
    import osqp
    _OSQP_AVAILABLE = True
except ImportError:
    _OSQP_AVAILABLE = False

_MCU_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class LinearMPC:
    """
    Condensed linear MPC with warm-starting OSQP.

    Lifts N steps of the linear model into a single QP:
        min  0.5 * U' H U + f' U
        s.t. lb <= U <= ub          (box constraints on each u_k)

    where U = [u_0; u_1; ... u_{N-1}], shape (N*nu,).

    The OSQP instance is created once and updated in-place on each model
    change, so the previous solution is always available as a warm start.
    Q_bar and R_bar are computed once at construction (they never change).
    """

    def __init__(self, A, B, Q, R, N, u_min, u_max):
        nx, nu = B.shape
        self._nx = nx
        self._nu = nu
        self._N = N
        self._A = A
        self._B = B
        self._Q = Q
        self._R = R

        # Box constraints: lb <= U <= ub
        self._lb = np.tile(u_min, N)
        self._ub = np.tile(u_max, N)

        # Q_bar, R_bar never change — compute once
        self._Q_bar = np.kron(np.eye(N), Q)
        self._R_bar = np.kron(np.eye(N), R)

        self._solver = None
        self._osqp_H_upper = None  # stores the sparse upper-tri H used in setup
        self.last_status = 'init'  # last solve outcome — readable by the node
        self._build_problem()

    # ── Internal builders ────────────────────────────────────────────────────

    def _build_problem(self):
        """Rebuild condensed prediction matrices and refresh the QP Hessian."""
        A, B, N, nx, nu = self._A, self._B, self._N, self._nx, self._nu

        # Precompute A^0 .. A^N to avoid repeated matrix_power calls
        A_pows = [np.eye(nx)]
        for _ in range(N):
            A_pows.append(A @ A_pows[-1])

        # S_x (N*nx, nx): S_x[k] = A^(k+1)  maps x0 → predicted states
        S_x = np.vstack([A_pows[k + 1] for k in range(N)])

        # S_u (N*nx, N*nu): S_u[k,j] = A^(k-j) @ B  maps U → predicted states
        S_u = np.zeros((N * nx, N * nu))
        for k in range(N):
            for j in range(k + 1):
                S_u[k * nx:(k + 1) * nx, j * nu:(j + 1) * nu] = A_pows[k - j] @ B

        # QP Hessian: H = S_u' Q_bar S_u + R_bar
        H = S_u.T @ self._Q_bar @ S_u + self._R_bar
        self._H = (H + H.T) / 2  # guarantee numerical symmetry
        self._S_u = S_u
        self._S_x = S_x

        if _OSQP_AVAILABLE:
            self._update_or_setup_osqp()

    def _update_or_setup_osqp(self):
        """Create the OSQP solver on first call; update H in-place on subsequent calls."""
        if self._solver is None:
            self._setup_osqp()
        else:
            self._refresh_osqp_h()

    def _setup_osqp(self):
        """Create a fresh OSQP instance (called once at startup)."""
        N_u = self._N * self._nu
        # OSQP only needs the upper triangle of the symmetric H
        H_upper = sp.triu(sp.csc_matrix(self._H), format='csc')
        I_sp = sp.eye(N_u, format='csc')
        self._osqp_H_upper = H_upper
        self._solver = osqp.OSQP()
        self._solver.setup(
            H_upper, np.zeros(N_u),
            I_sp, self._lb, self._ub,
            warm_starting=True,
            verbose=False,
            eps_abs=1e-4,
            eps_rel=1e-4,
            max_iter=500,
            polish=False,
        )

    def _refresh_osqp_h(self):
        """
        Update the Hessian inside the existing OSQP instance without recreating it.

        OSQP's update(Px=...) requires the same sparsity structure (same indices)
        as the original setup call.  If the structure has changed (rare, but
        possible when interaction-matrix entries are exactly zero at equilibrium),
        we fall back to a full solver rebuild.
        """
        H_upper_new = sp.triu(sp.csc_matrix(self._H), format='csc')
        old = self._osqp_H_upper
        structure_changed = (
            H_upper_new.nnz != old.nnz
            or not np.array_equal(H_upper_new.indices, old.indices)
            or not np.array_equal(H_upper_new.indptr, old.indptr)
        )
        if structure_changed:
            # Sparsity pattern changed — rebuild solver (keeps warm-start on next step)
            self._setup_osqp()
            return
        self._solver.update(Px=H_upper_new.data)
        self._osqp_H_upper = H_upper_new

    # ── Public interface ──────────────────────────────────────────────────────

    def update_model(self, A, B):
        """Update A, B and refresh all QP data.  OSQP warm-start is preserved."""
        self._A = A
        self._B = B
        self._build_problem()

    def solve(self, x0):
        """Return optimal u_0* given initial state x0 (shape (nx,))."""
        f = self._S_u.T @ self._Q_bar @ (self._S_x @ x0)

        if self._solver is not None:
            self._solver.update(q=f)
            result = self._solver.solve()
            if result.info.status in ('solved', 'solved_inaccurate'):
                self.last_status = result.info.status
                return result.x[:self._nu]
            self.last_status = 'osqp_failed'
        else:
            self.last_status = 'numpy_fallback'

        # Unconstrained fallback: U* = -H^{-1} f, then clip to box bounds.
        try:
            U = np.linalg.solve(self._H, -f)
        except np.linalg.LinAlgError:
            U = np.linalg.lstsq(self._H, -f, rcond=None)[0]
        U = np.clip(U, self._lb, self._ub)
        return U[:self._nu]


# ── Interaction matrix and augmented model ───────────────────────────────────

def _build_interaction_matrix(fx, fy, Z, desired_area, eu=0.0, ev=0.0):
    """
    Build the 3×2 interaction matrix L at the current operating point.

    Camera frame: z_cam = forward, x_cam = right, y_cam = down.
    Robot inputs:  v (forward m/s), omega (yaw rad/s, CCW positive).

    Camera velocity from robot motion:
        vz_cam = v,   wy_cam = -omega   (CCW yaw → CW rotation about y_cam_down)

    Standard IBVS point interaction (Chaumette 2006), mapped to pixel errors
    eu = fx*xn, ev = fy*yn:

        d_eu/dt = (eu/Z)*v + fx*(1 + xn^2)*omega
        d_ev/dt = (ev/Z)*v + fy*(xn*yn)*omega

    For the area feature ea = sqrt(area) - sqrt(desired_area):
        sqrt(area) = sqrt(pi)*fx*r / Z,  so d(sqrt(A))/dt = sqrt(A)/Z * vz.
        dZ/dt = -vz  (depth decreases as camera approaches), therefore:
        d(1/Z)/dt = +vz/Z^2  →  d_ea/dt = +sqrt(A_des)/Z * v   (POSITIVE)
        Forward motion increases projected area — this is the correct sign.

    Returns L (3×2) mapping [v, omega] → [d_eu/dt, d_ev/dt, d_ea/dt].
    """
    sqrt_A_des = np.sqrt(desired_area)
    xn = eu / fx if fx != 0.0 else 0.0
    yn = ev / fy if fy != 0.0 else 0.0

    L = np.array([
        [eu / Z,              fx * (1.0 + xn * xn)],
        [ev / Z,              fy * (xn * yn)       ],
        [sqrt_A_des / Z,      0.0                  ],
    ])
    return L


def _build_augmented_model(L: np.ndarray, Ts: float,
                           tau_v: float, tau_o: float):
    """
    Build the 5×5 discrete-time augmented model x_{k+1} = A*x_k + B*u_k.

    State:   x = [eu, ev, ea, v, omega]
    Control: u = [v_cmd, omega_cmd]

    Image states are driven by *actual* velocities (not commands) — this
    separates the actuation lag from the visual dynamics.

    Velocity dynamics are first-order ZOH (identified from step-response):
        v_{k+1}     = (1 - Ts/tau_v)*v_k     + (Ts/tau_v)*v_cmd_k
        omega_{k+1} = (1 - Ts/tau_o)*omega_k + (Ts/tau_o)*omega_cmd_k
    """
    alpha_v = Ts / tau_v
    alpha_o = Ts / tau_o

    A = np.array([
        [1., 0., 0., Ts * L[0, 0], Ts * L[0, 1]],
        [0., 1., 0., Ts * L[1, 0], Ts * L[1, 1]],
        [0., 0., 1., Ts * L[2, 0], 0.           ],
        [0., 0., 0., 1. - alpha_v, 0.            ],
        [0., 0., 0., 0.,           1. - alpha_o  ],
    ])
    B = np.array([
        [0.,      0.     ],
        [0.,      0.     ],
        [0.,      0.     ],
        [alpha_v, 0.     ],
        [0.,      alpha_o],
    ])
    return A, B


def _estimate_depth(ea, desired_area, Z_star,
                    z_min: float = 0.10, z_max: float = 3.0) -> float:
    """
    Recover current depth from the measured area error.

    From the feature definition:  ea = sqrt(area) - sqrt(desired_area)
    => sqrt(area) = ea + sqrt(desired_area)
    => area = (ea + sqrt(desired_area))^2

    From the projection model:  sqrt(area) = C / Z  (C = sqrt(pi)*fx*r)
    => Z = Z_star * sqrt(desired_area) / sqrt(area)  = Z_star * sqrt(desired_area / area)

    Clamped to [z_min, z_max] to guard against degenerate detections.
    """
    sqrt_area = ea + np.sqrt(desired_area)
    if sqrt_area <= 0.0:
        return Z_star
    area = sqrt_area ** 2
    return float(np.clip(Z_star * np.sqrt(desired_area / area), z_min, z_max))


# ── ROS 2 node ────────────────────────────────────────────────────────────────

class MPCIBVSNode(Node):

    def __init__(self):
        super().__init__('mpc_ibvs_node')

        self.declare_parameter('Ts', 0.08)
        self.declare_parameter('N', 15)
        self.declare_parameter('Q_diag', [1.0, 0.0, 0.8])   # ev (index 1) un-penalized
        self.declare_parameter('R_diag', [0.40, 1.00])
        self.declare_parameter('v_max', 0.30)
        self.declare_parameter('v_min', -0.05)
        self.declare_parameter('omega_max', 1.20)
        self.declare_parameter('lost_timeout_s', 0.5)
        self.declare_parameter('fx', 486.2936)
        self.declare_parameter('fy', 488.4081)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 240.0)
        self.declare_parameter('nominal_depth_Z', 0.5)
        self.declare_parameter('desired_area', 8000.0)
        self.declare_parameter('online_linearization', True)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('tau_v',     0.40)
        self.declare_parameter('tau_omega', 0.40)
        self.declare_parameter('dead_zone_eu',       20.0)
        self.declare_parameter('dead_zone_ev',   100000.0)  # disabled — ev not controlled
        self.declare_parameter('dead_zone_ea',       30.0)
        self.declare_parameter('dead_zone_exit_factor', 3.0)
        self.declare_parameter('dead_zone_persist_frames', 3)
        self.declare_parameter('lost_decel_s', 0.4)
        self.declare_parameter('diag_period_s', 1.0)
        # Feature-error normalization scales. Q_diag is applied to the *normalized*
        # errors eu/feat_norm_uv, ev/feat_norm_uv, ea/feat_norm_a, so Q and R end
        # up on comparable magnitudes. Without this the pixel-scaled tracking cost
        # (∝ error², i.e. hundreds²) dwarfs R and the QP degenerates into a
        # deadbeat controller that bang-bangs v/ω to their limits every cycle.
        self.declare_parameter('feat_norm_uv', 600.0)   # ≈ half-width of the image, px
        self.declare_parameter('feat_norm_a',   50.0)   # ≈ sqrt(area) dynamic range, px

        Ts          = self.get_parameter('Ts').value
        N           = self.get_parameter('N').value
        Q_diag      = self.get_parameter('Q_diag').value
        R_diag      = self.get_parameter('R_diag').value
        v_max       = self.get_parameter('v_max').value
        v_min       = self.get_parameter('v_min').value
        omega_max   = self.get_parameter('omega_max').value
        fx          = self.get_parameter('fx').value
        fy          = self.get_parameter('fy').value
        Z_star      = self.get_parameter('nominal_depth_Z').value
        desired_area = float(self.get_parameter('desired_area').value)

        self._online_linearization = bool(self.get_parameter('online_linearization').value)
        self._conf_threshold = float(self.get_parameter('confidence_threshold').value)
        self._lost_timeout = self.get_parameter('lost_timeout_s').value
        self._v_max = v_max
        self._v_min = v_min
        self._omega_max = omega_max
        self._Ts = Ts
        self._fx = fx
        self._fy = fy
        self._Z_star = Z_star
        self._desired_area = desired_area
        self._tau_v = self.get_parameter('tau_v').value
        self._tau_omega = self.get_parameter('tau_omega').value
        self._dz_eu = float(self.get_parameter('dead_zone_eu').value)
        self._dz_ev = float(self.get_parameter('dead_zone_ev').value)
        self._dz_ea = float(self.get_parameter('dead_zone_ea').value)
        self._dz_exit_factor = float(self.get_parameter('dead_zone_exit_factor').value)
        self._dz_persist = int(self.get_parameter('dead_zone_persist_frames').value)
        self._lost_decel_s = float(self.get_parameter('lost_decel_s').value)
        self._diag_period_s = float(self.get_parameter('diag_period_s').value)
        feat_norm_uv = float(self.get_parameter('feat_norm_uv').value)
        feat_norm_a  = float(self.get_parameter('feat_norm_a').value)

        # Initial model at equilibrium (eu=0, ev=0, Z=Z_star)
        L_star = _build_interaction_matrix(fx, fy, Z_star, desired_area)
        A, B = _build_augmented_model(L_star, Ts, self._tau_v, self._tau_omega)

        # Q is 5×5: penalize image features only; v/omega are prediction states.
        # Scale by 1/feat_norm² so Q acts on normalized errors (see param doc).
        feat_norm_sq = [feat_norm_uv ** 2, feat_norm_uv ** 2, feat_norm_a ** 2]
        Q_diag_scaled = [q / nsq for q, nsq in zip(Q_diag, feat_norm_sq)]
        Q_diag_full = Q_diag_scaled + [0.0, 0.0]
        Q = np.diag(Q_diag_full)
        R = np.diag(R_diag)
        u_min = np.array([v_min, -omega_max])
        u_max = np.array([v_max,  omega_max])

        self._mpc = LinearMPC(A, B, Q, R, N, u_min, u_max)

        if not _OSQP_AVAILABLE:
            self.get_logger().warn(
                'osqp not found — using unconstrained numpy fallback. '
                'Install with: pip install osqp'
            )

        self._feat           = None   # latest [eu, ev, ea, conf]
        self._last_feat_time = None
        self._v_meas = 0.0         # actual linear velocity from /robot_vel [m/s]
        self._omega_meas = 0.0     # actual angular velocity from /robot_vel [rad/s]
        self._in_dead_zone = False  # hysteresis state for dead zone
        self._dz_entry_count = 0   # consecutive frames inside entry threshold
        self._dz_exit_count  = 0   # consecutive frames outside exit threshold

        # Soft-stop state: ramp toward zero over `lost_decel_s` instead of stepping
        self._last_cmd_v = 0.0
        self._last_cmd_omega = 0.0

        # Per-branch counters for the 1 Hz diag log
        self._n_servo = 0
        self._n_zero_lost = 0
        self._n_zero_low_conf = 0
        self._n_zero_dead_zone = 0
        self._diag_conf_samples = []
        self._diag_age_samples = []

        self.create_subscription(
            Float64MultiArray, '/visual_features', self._feat_callback, 10)
        self.create_subscription(
            TwistStamped, '/robot_vel', self._robot_vel_callback, _MCU_QOS)

        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel_desired', 10)

        # Debug topic — published every cycle when at least one subscriber exists.
        # Float64MultiArray indices:
        #   [0]  eu           pixel error horizontal    [px]
        #   [1]  ev           pixel error vertical      [px]
        #   [2]  ea           scale/depth error         [sqrt(px²)]
        #   [3]  v_meas       measured linear velocity  [m/s]
        #   [4]  omega_meas   measured angular velocity [rad/s]
        #   [5]  v_cmd        commanded linear velocity [m/s]
        #   [6]  omega_cmd    commanded angular velocity[rad/s]
        #   [7]  Z_est        estimated depth           [m]
        #   [8]  solve_ms     QP solve time             [ms]
        #   [9]  feat_age_ms  age of last feature msg   [ms]
        #   [10] status       2=solved 1=inaccurate 0=numpy_fallback
        #                     -1=osqp_failed -2=safety_stop
        self._dbg_pub = self.create_publisher(Float64MultiArray, '/mpc_ibvs/debug', 1)

        self.create_timer(Ts, self._control_loop)
        self.create_timer(self._diag_period_s, self._publish_diag_summary)

        self.get_logger().info(
            f'MPCIBVSNode ready — N={N}, Ts={Ts}s, '
            f'v=[{v_min:.2f},{v_max:.2f}], omega_max={omega_max:.2f}, '
            f'tau_v={self._tau_v:.3f}s, tau_omega={self._tau_omega:.3f}s, '
            f'OSQP={_OSQP_AVAILABLE}'
        )
        self.get_logger().info(
            f'Interaction matrix at equilibrium L*:\n{np.array2string(L_star, precision=4)}'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _feat_callback(self, msg: Float64MultiArray):
        if len(msg.data) < 4:
            self.get_logger().warn('Malformed /visual_features — expected 4 values.')
            return
        self._feat = msg.data
        self._last_feat_time = self.get_clock().now()

    def _robot_vel_callback(self, msg: TwistStamped):
        self._v_meas     = msg.twist.linear.x
        self._omega_meas = msg.twist.angular.z

    # ── Control loop ──────────────────────────────────────────────────────────

    def _control_loop(self):
        now = self.get_clock().now()
        target_v, target_omega = 0.0, 0.0   # default: decelerate to stop

        # Determine whether to servo. Also categorize the no-servo reason so the
        # 1 Hz diag log can tell the operator *why* the MPC was silent.
        servo = False
        no_servo_reason = 'lost'  # default for "no feature ever received"
        feat_age = None
        feat_conf = None
        if self._feat is not None and self._last_feat_time is not None:
            feat_age = (now - self._last_feat_time).nanoseconds * 1e-9
            if feat_age > self._lost_timeout:
                self._in_dead_zone = False  # reset so we re-acquire cleanly
                self._dz_entry_count = 0
                self._dz_exit_count  = 0
                no_servo_reason = 'lost'
                if feat_age < self._lost_timeout + 1.0:  # log once per second
                    self.get_logger().warn(
                        f'Visual features stale ({feat_age:.2f}s > {self._lost_timeout}s) — stopping.'
                    )
            else:
                eu, ev, ea, conf = self._feat
                feat_conf = float(conf)
                if conf >= self._conf_threshold:
                    # Dead zone with persistence hysteresis.
                    in_thresh = (abs(eu) < self._dz_eu and
                                 abs(ev) < self._dz_ev and
                                 abs(ea) < self._dz_ea)
                    out_thresh = (abs(eu) > self._dz_eu * self._dz_exit_factor or
                                  abs(ev) > self._dz_ev * self._dz_exit_factor or
                                  abs(ea) > self._dz_ea * self._dz_exit_factor)

                    if self._in_dead_zone:
                        if out_thresh:
                            self._dz_exit_count += 1
                            self._dz_entry_count = 0
                            if self._dz_exit_count >= self._dz_persist:
                                self._in_dead_zone = False
                                self._dz_exit_count = 0
                        else:
                            self._dz_exit_count = 0
                    else:
                        if in_thresh:
                            self._dz_entry_count += 1
                            self._dz_exit_count = 0
                            if self._dz_entry_count >= self._dz_persist:
                                self._in_dead_zone = True
                                self._dz_entry_count = 0
                        else:
                            self._dz_entry_count = 0

                    servo = not self._in_dead_zone
                    if not servo:
                        no_servo_reason = 'dead_zone'
                else:
                    no_servo_reason = 'low_conf'

        # Record diag samples for this cycle
        if feat_conf is not None:
            self._diag_conf_samples.append(feat_conf)
        if feat_age is not None:
            self._diag_age_samples.append(feat_age)

        # MPC solve (only when actively servoing)
        Z_est = self._Z_star
        dt_solve = 0.0
        feat_age_ms = 0.0
        eu = ev = ea = 0.0

        if servo:
            self._n_servo += 1
            eu, ev, ea, conf = self._feat
            feat_age_ms = feat_age * 1e3 if feat_age is not None else 0.0
            Z_est = _estimate_depth(ea, self._desired_area, self._Z_star)
            if self._online_linearization:
                L_cur = _build_interaction_matrix(
                    self._fx, self._fy, Z_est, self._desired_area,
                    eu=float(eu), ev=float(ev),
                )
                A_aug, B_aug = _build_augmented_model(
                    L_cur, self._Ts, self._tau_v, self._tau_omega
                )
                self._mpc.update_model(A_aug, B_aug)

            x0 = np.array([eu, ev, ea, self._v_meas, self._omega_meas])
            t0 = time.monotonic()
            u_opt = self._mpc.solve(x0)
            dt_solve = (time.monotonic() - t0) * 1e3

            target_v     = _clamp(float(u_opt[0]), self._v_min, self._v_max)
            target_omega = _clamp(float(u_opt[1]), -self._omega_max, self._omega_max)

            if dt_solve > 10.0:
                self.get_logger().warn(f'MPC solve took {dt_solve:.1f} ms')
        else:
            if no_servo_reason == 'lost':
                self._n_zero_lost += 1
            elif no_servo_reason == 'low_conf':
                self._n_zero_low_conf += 1
            else:
                self._n_zero_dead_zone += 1

            if self._lost_decel_s > 0.0:
                decay = self._Ts / self._lost_decel_s
                target_v     = self._last_cmd_v     * max(0.0, 1.0 - decay)
                target_omega = self._last_cmd_omega * max(0.0, 1.0 - decay)
                if abs(target_v) < 1e-3:
                    target_v = 0.0
                if abs(target_omega) < 1e-3:
                    target_omega = 0.0

        self._last_cmd_v = target_v
        self._last_cmd_omega = target_omega

        msg = Twist()
        msg.linear.x = target_v
        msg.angular.z = target_omega
        self._cmd_pub.publish(msg)

        _STATUS = {'solved': 2.0, 'solved_inaccurate': 1.0,
                   'numpy_fallback': 0.0, 'osqp_failed': -1.0}
        self._publish_debug(eu, ev, ea, target_v, target_omega,
                            Z_est, dt_solve, feat_age_ms,
                            _STATUS.get(self._mpc.last_status, -2.0))

    def _publish_diag_summary(self):
        # Per-second snapshot of which branch dominated. If `lost` dominates the
        # feature stream is starving; `low_conf` means the detector is unsure;
        # `dead_zone` means the robot thinks it has converged.
        total = (self._n_servo + self._n_zero_lost +
                 self._n_zero_low_conf + self._n_zero_dead_zone)
        conf_str = age_str = '—'
        if self._diag_conf_samples:
            arr = np.asarray(self._diag_conf_samples)
            conf_str = f'{arr.min():.2f}/{arr.mean():.2f}/{arr.max():.2f}'
        if self._diag_age_samples:
            arr = np.asarray(self._diag_age_samples)
            age_str = f'{arr.min():.3f}/{arr.mean():.3f}/{arr.max():.3f}'

        self.get_logger().info(
            f'MPC diag: total={total} servo={self._n_servo} '
            f'zero_lost={self._n_zero_lost} zero_low_conf={self._n_zero_low_conf} '
            f'zero_dead={self._n_zero_dead_zone} '
            f'conf[min/avg/max]={conf_str} age[min/avg/max]={age_str}'
        )

        self._n_servo = 0
        self._n_zero_lost = 0
        self._n_zero_low_conf = 0
        self._n_zero_dead_zone = 0
        self._diag_conf_samples.clear()
        self._diag_age_samples.clear()

    def _publish_debug(self, eu, ev, ea, v_cmd, omega_cmd,
                        Z_est, solve_ms, feat_age_ms, status_code):
        if self._dbg_pub.get_subscription_count() == 0:
            return
        msg = Float64MultiArray()
        msg.data = [
            float(eu), float(ev), float(ea),
            self._v_meas, self._omega_meas,
            float(v_cmd), float(omega_cmd),
            float(Z_est),
            float(solve_ms),
            float(feat_age_ms),
            float(status_code),
        ]
        self._dbg_pub.publish(msg)

    def _publish_stop(self):
        self._cmd_pub.publish(Twist())
        # Publish debug so you can see what triggered the stop even when stopped
        age_ms = 0.0
        if self._feat is not None and self._last_feat_time is not None:
            age_ms = (self.get_clock().now() - self._last_feat_time).nanoseconds * 1e-6
        eu, ev, ea = (self._feat[0], self._feat[1], self._feat[2]) if self._feat else (0., 0., 0.)
        self._publish_debug(eu, ev, ea, 0.0, 0.0, self._Z_star, 0.0, age_ms, -2.0)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = MPCIBVSNode()

    def _shutdown(signum, frame):
        node._publish_stop()
        rclpy.try_shutdown()

    signal.signal(signal.SIGTERM, _shutdown)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node._publish_stop()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
