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

Linearisation of L is refreshed each cycle from current (eu, ev) — LTV-MPC.
The QP is condensed over the horizon N and solved with OSQP.

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
    Condensed linear MPC.

    Lifts N steps of the linear model into a single QP:
        min  0.5 * U' H U + f' U
        s.t. lb <= U <= ub          (box constraints on each u_k)

    where U = [u_0; u_1; ... u_{N-1}], shape (N*nu,).
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

        self._solver = None
        self._build_problem()

    def _build_problem(self):
        """Build condensed prediction and QP matrices for current A, B."""
        A = self._A
        B = self._B
        N = self._N
        nx = self._nx
        nu = self._nu

        # Build prediction matrices: S_x (N*nx x nx), S_u (N*nx x N*nu)
        S_x = np.zeros((N * nx, nx))
        S_u = np.zeros((N * nx, N * nu))
        Ak = np.eye(nx)
        for k in range(N):
            Ak = A @ Ak
            S_x[k * nx:(k + 1) * nx, :] = Ak
            for j in range(k + 1):
                S_u[k * nx:(k + 1) * nx, j * nu:(j + 1) * nu] = (
                    np.linalg.matrix_power(A, k - j) @ B
                )

        # Block-diagonal cost matrices over horizon
        Q_bar = np.kron(np.eye(N), self._Q)
        R_bar = np.kron(np.eye(N), self._R)

        # QP Hessian: H = S_u' Q_bar S_u + R_bar
        self._H = S_u.T @ Q_bar @ S_u + R_bar
        self._H = (self._H + self._H.T) / 2  # ensure symmetry
        self._S_u = S_u
        self._S_x = S_x
        self._Q_bar = Q_bar

        if _OSQP_AVAILABLE:
            self._setup_osqp()

    def update_model(self, A, B):
        """Update model matrices and rebuild the condensed QP data."""
        self._A = A
        self._B = B
        self._build_problem()

    def _setup_osqp(self):
        N_u = self._N * self._nu
        H_sp = sp.csc_matrix(self._H)
        # Identity constraint matrix for box bounds
        I_sp = sp.eye(N_u, format='csc')
        self._solver = osqp.OSQP()
        self._solver.setup(
            H_sp, np.zeros(N_u),
            I_sp, self._lb, self._ub,
            warm_starting=True,
            verbose=False,
            eps_abs=1e-4,
            eps_rel=1e-4,
            max_iter=500,
            polish=False,
        )

    def solve(self, x0):
        """Return optimal u_0* given initial state x0 (shape (nx,))."""
        f = self._S_u.T @ self._Q_bar @ (self._S_x @ x0)

        if self._solver is not None:
            self._solver.update(q=f)
            result = self._solver.solve()
            if result.info.status in ('solved', 'solved_inaccurate'):
                return result.x[:self._nu]
            # Fall through to numpy fallback on solver failure

        # Unconstrained fallback: U* = -H^{-1} f, clipped to bounds.
        # Use least squares if H is singular/ill-conditioned.
        try:
            U = np.linalg.solve(self._H, -f)
        except np.linalg.LinAlgError:
            U = np.linalg.lstsq(self._H, -f, rcond=None)[0]
        U = np.clip(U, self._lb, self._ub)
        return U[:self._nu]


def _build_interaction_matrix(fx, fy, Z_star, desired_area, eu=0.0, ev=0.0):
    """
    Build a local 3x2 interaction matrix L evaluated at current feature error
    (eu, ev) and depth Z_star.

    Camera frame convention: z_cam = forward, x_cam = right, y_cam = down.
    Robot control inputs: [v (forward), omega (yaw, CCW positive)].

    Camera velocity induced by robot:
        v_cam  = [0, 0, v]     (forward translation)
        omega_cam = [0, -omega, 0]  (yaw: robot CCW -> camera rotates CW about y_cam)

    Standard interaction matrix for a point at normalised coords (xn, yn), depth Z:
        d/dt [xn] = [xn*vz/Z - (1+xn^2)*wy + yn*wz - vx/Z]
        d/dt [yn] = [yn*vz/Z - xn*yn*wy - xn*wz - vy/Z]

        With pixel errors eu = fx*xn, ev = fy*yn, and wy = -omega:
            d_eu/dt = (eu/Z)*v + fx*(1 + xn^2)*omega
            d_ev/dt = (ev/Z)*v + fy*(xn*yn)*omega

        At s* = 0 this simplifies to:
            d_eu/dt = fx*omega, d_ev/dt = 0.
    Area of a projected circle: A = pi*(fx*r/Z)^2, so sqrt(A) = sqrt(pi)*fx*r/Z.
    d/dt sqrt(A) = -sqrt(pi)*fx*r/Z^2 * vz = -sqrt(A_des)/Z * v

    Returns L (3x2) mapping [v, omega] to [d_eu/dt, d_ev/dt, d_ea/dt].
    """
    sqrt_A_des = np.sqrt(desired_area)
    xn = eu / fx if fx != 0.0 else 0.0
    yn = ev / fy if fy != 0.0 else 0.0

    # Positive omega (robot CCW) pushes the target right in the image.
    # The optimizer applies opposite-signed omega to reduce eu.

    L = np.array([
        [eu / Z_star,   fx * (1.0 + xn * xn)],
        [ev / Z_star,   fy * (xn * yn)],
        [-sqrt_A_des / Z_star, 0.0],  # d_ea/dt = -sqrt_A_des/Z * v
    ])
    return L


def _build_augmented_model(L: np.ndarray, Ts: float,
                           tau_v: float, tau_o: float):
    """
    Build 5×5 augmented state-space model with first-order velocity dynamics.

    State:   x = [eu, ev, ea, v, omega]  (5,)
    Control: u = [v_cmd, omega_cmd]      (2,)

    Image dynamics (driven by actual velocity states, not commands directly):
        eu_{k+1}    = eu_k + Ts*(L[0,0]*v_k + L[0,1]*omega_k)
        ev_{k+1}    = ev_k + Ts*(L[1,0]*v_k + L[1,1]*omega_k)
        ea_{k+1}    = ea_k + Ts*(L[2,0]*v_k)

    Closed-loop velocity dynamics (first-order ZOH, captures actuator lag):
        v_{k+1}     = (1 - Ts/tau_v)*v_k     + (Ts/tau_v)*v_cmd_k
        omega_{k+1} = (1 - Ts/tau_o)*omega_k + (Ts/tau_o)*omega_cmd_k

    tau_v and tau_o are the closed-loop step-response time constants of the
    full velocity cascade (Jetson PI + MCU PID). Identify via step response test.

    Returns A (5x5), B (5x2).
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


class MPCIBVSNode(Node):

    def __init__(self):
        super().__init__('mpc_ibvs_node')

        self.declare_parameter('Ts', 0.08)
        self.declare_parameter('N', 15)
        self.declare_parameter('Q_diag', [1.0, 1.0, 0.5])
        self.declare_parameter('R_diag', [0.1, 0.2])
        self.declare_parameter('v_max', 0.30)
        self.declare_parameter('v_min', -0.05)
        self.declare_parameter('omega_max', 1.20)
        self.declare_parameter('lost_timeout_s', 0.5)
        self.declare_parameter('fx', 640.0)
        self.declare_parameter('fy', 640.0)
        self.declare_parameter('cx', 640.0)
        self.declare_parameter('cy', 360.0)
        self.declare_parameter('nominal_depth_Z', 0.5)
        self.declare_parameter('desired_area', 8000.0)
        self.declare_parameter('online_linearization', True)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('tau_v',     0.15)
        self.declare_parameter('tau_omega', 0.10)

        Ts = self.get_parameter('Ts').value
        N = self.get_parameter('N').value
        Q_diag = self.get_parameter('Q_diag').value
        R_diag = self.get_parameter('R_diag').value
        v_max = self.get_parameter('v_max').value
        v_min = self.get_parameter('v_min').value
        omega_max = self.get_parameter('omega_max').value

        fx = self.get_parameter('fx').value
        fy = self.get_parameter('fy').value
        Z_star = self.get_parameter('nominal_depth_Z').value
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

        L_star = _build_interaction_matrix(fx, fy, Z_star, desired_area)
        A, B = _build_augmented_model(L_star, Ts, self._tau_v, self._tau_omega)

        # Q is 5×5: penalize image features only; v/omega are prediction states
        Q_diag_full = list(Q_diag) + [0.0, 0.0]
        Q = np.diag(Q_diag_full)
        R = np.diag(R_diag)
        u_min = np.array([v_min, -omega_max])
        u_max = np.array([v_max, omega_max])

        self._mpc = LinearMPC(A, B, Q, R, N, u_min, u_max)

        if not _OSQP_AVAILABLE:
            self.get_logger().warn(
                'osqp not found — using unconstrained numpy fallback. '
                'Install with: pip install osqp'
            )

        self._feat = None          # latest [eu, ev, ea, conf]
        self._last_feat_time = None
        self._v_meas = 0.0         # actual linear velocity from /robot_vel [m/s]
        self._omega_meas = 0.0     # actual angular velocity from /robot_vel [rad/s]

        self.create_subscription(
            Float64MultiArray,
            '/visual_features',
            self._feat_callback,
            10,
        )
        self.create_subscription(
            TwistStamped,
            '/robot_vel',
            self._robot_vel_callback,
            _MCU_QOS,
        )

        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel_desired', 10)

        self.create_timer(Ts, self._control_loop)

        self.get_logger().info(
            f'MPCIBVSNode ready — N={N}, Ts={Ts}s, '
            f'v=[{v_min:.2f},{v_max:.2f}], omega_max={omega_max:.2f}, '
            f'tau_v={self._tau_v:.3f}s, tau_omega={self._tau_omega:.3f}s, '
            f'OSQP={_OSQP_AVAILABLE}'
        )
        self.get_logger().info(f'Interaction matrix L_s*:\n{np.array2string(L_star, precision=4)}')

    def _feat_callback(self, msg: Float64MultiArray):
        if len(msg.data) < 4:
            self.get_logger().warn('Received malformed /visual_features (expected 4 values).')
            return
        self._feat = msg.data
        self._last_feat_time = self.get_clock().now()

    def _robot_vel_callback(self, msg: TwistStamped):
        self._v_meas = msg.twist.linear.x
        self._omega_meas = msg.twist.angular.z

    def _control_loop(self):
        now = self.get_clock().now()

        # Safety stop if detection is lost or stale
        if self._feat is None or self._last_feat_time is None:
            self._publish_stop()
            return

        age = (now - self._last_feat_time).nanoseconds * 1e-9
        if age > self._lost_timeout:
            if age < self._lost_timeout + 1.0:  # log once per second
                self.get_logger().warn(
                    f'Visual features stale ({age:.2f}s > {self._lost_timeout}s) — stopping.'
                )
            self._publish_stop()
            return

        eu, ev, ea, conf = self._feat

        if conf < self._conf_threshold:
            self._publish_stop()
            return

        if self._online_linearization:
            L_star = _build_interaction_matrix(
                self._fx,
                self._fy,
                self._Z_star,
                self._desired_area,
                eu=float(eu),
                ev=float(ev),
            )
            A_aug, B_aug = _build_augmented_model(
                L_star, self._Ts, self._tau_v, self._tau_omega
            )
            self._mpc.update_model(A_aug, B_aug)

        # Augmented initial state: image features + actual body velocities
        x0 = np.array([eu, ev, ea, self._v_meas, self._omega_meas])
        t0 = time.monotonic()
        u_opt = self._mpc.solve(x0)
        dt_solve = (time.monotonic() - t0) * 1e3

        v = _clamp(float(u_opt[0]), self._v_min, self._v_max)
        omega = _clamp(float(u_opt[1]), -self._omega_max, self._omega_max)

        msg = Twist()
        msg.linear.x = v
        msg.angular.z = omega
        self._cmd_pub.publish(msg)

        if dt_solve > 10.0:
            self.get_logger().warn(f'MPC solve took {dt_solve:.1f} ms')

    def _publish_stop(self):
        self._cmd_pub.publish(Twist())


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
