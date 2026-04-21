#!/usr/bin/env python3
"""
Linear MPC node for Image-Based Visual Servoing (IBVS).

State:    s = [eu, ev, ea]   (pixel centroid errors + scale error)
Control:  u = [v, omega]     (linear and angular velocity)

The prediction model is a linearisation of the interaction matrix at s* = 0:

    s_{k+1} = A * s_k + B * u_k
    A = I_3
    B = Ts * L_s*             (3x2, built from camera intrinsics at goal depth)

The QP is condensed over the horizon N and solved with OSQP.

Subscribes:
  /visual_features  (std_msgs/Float64MultiArray)  [eu, ev, ea, confidence]

Publishes:
  /cmd_vel_desired  (geometry_msgs/Twist)
"""

import signal
import time
import numpy as np
import scipy.sparse as sp

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

try:
    import osqp
    _OSQP_AVAILABLE = True
except ImportError:
    _OSQP_AVAILABLE = False


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
        Q_bar = np.kron(np.eye(N), Q)
        R_bar = np.kron(np.eye(N), R)

        # QP Hessian: H = S_u' Q_bar S_u + R_bar
        self._H = S_u.T @ Q_bar @ S_u + R_bar
        self._H = (self._H + self._H.T) / 2  # ensure symmetry
        self._S_u = S_u
        self._S_x = S_x
        self._Q_bar = Q_bar

        # Box constraints: lb <= U <= ub
        self._lb = np.tile(u_min, N)
        self._ub = np.tile(u_max, N)

        self._solver = None
        if _OSQP_AVAILABLE:
            self._setup_osqp()

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

    def solve(self, s0):
        """Return optimal u_0* given initial state s0 (shape (nx,))."""
        f = self._S_u.T @ self._Q_bar @ (self._S_x @ s0)

        if self._solver is not None:
            self._solver.update(q=f)
            result = self._solver.solve()
            if result.info.status in ('solved', 'solved_inaccurate'):
                return result.x[:self._nu]
            # Fall through to numpy fallback on solver failure

        # Unconstrained fallback: U* = -H^{-1} f, clipped to bounds
        U = np.linalg.solve(self._H, -f)
        U = np.clip(U, self._lb, self._ub)
        return U[:self._nu]


def _build_interaction_matrix(fx, fy, Z_star, desired_area):
    """
    Build the 3x2 linearised interaction matrix L_s* evaluated at s* = 0
    (target centroid at image centre, at depth Z_star).

    Camera frame convention: z_cam = forward, x_cam = right, y_cam = down.
    Robot control inputs: [v (forward), omega (yaw, CCW positive)].

    Camera velocity induced by robot:
        v_cam  = [0, 0, v]     (forward translation)
        omega_cam = [0, -omega, 0]  (yaw: robot CCW -> camera rotates CW about y_cam)

    Standard interaction matrix for a point at normalised coords (xn, yn), depth Z:
        d/dt [xn] = [xn*vz/Z - (1+xn^2)*wy + yn*wz - vx/Z]
        d/dt [yn] = [yn*vz/Z - xn*yn*wy - xn*wz - vy/Z]

    At s* = 0: xn = yn = 0. Pixel errors: eu = fx*xn, ev = fy*yn.
    Area of a projected circle: A = pi*(fx*r/Z)^2, so sqrt(A) = sqrt(pi)*fx*r/Z.
    d/dt sqrt(A) = -sqrt(pi)*fx*r/Z^2 * vz = -sqrt(A_des)/Z * v

    Returns L (3x2) mapping [v, omega] to [d_eu/dt, d_ev/dt, d_ea/dt].
    """
    sqrt_A_des = np.sqrt(desired_area)

    # At xn=yn=0, vx=vy=0, vz=v, wz=0, wy=-omega:
    #   d_xn/dt = -(-omega)*(1+0) = omega  →  d_eu/dt = fx * omega * (-1) ... let's be careful

    # ẋn = xn*v/Z - (1+xn^2)*(-omega) = xn*v/Z + omega  (at xn=0: omega)
    # ẏn = yn*v/Z - xn*yn*(-omega)                        (at yn=0, xn=0: 0)
    # But we want d_eu/dt = fx * ẋn, d_ev/dt = fy * ẏn

    # So at s*=0:
    # d_eu/dt = fx * (0 * v/Z + omega)   = fx * omega
    # d_ev/dt = fy * 0                   = 0
    # d_ea/dt = -sqrt_A_des / Z * v

    # Note: omega here is in rad/s. A positive omega (CCW) moves the target
    # to the right in the image (positive eu), so the controller will apply
    # negative omega to bring eu back to zero — which is correct.

    L = np.array([
        [0.0,           fx],        # d_eu/dt = fx * omega
        [0.0,           0.0],       # d_ev/dt ≈ 0 (zero at linearisation point)
        [-sqrt_A_des / Z_star, 0.0],  # d_ea/dt = -sqrt_A_des/Z * v
    ])
    return L


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

        self._lost_timeout = self.get_parameter('lost_timeout_s').value
        self._v_max = v_max
        self._v_min = v_min
        self._omega_max = omega_max

        L_star = _build_interaction_matrix(fx, fy, Z_star, desired_area)

        A = np.eye(3)
        B = Ts * L_star
        Q = np.diag(Q_diag)
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

        self.create_subscription(
            Float64MultiArray,
            '/visual_features',
            self._feat_callback,
            10,
        )

        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel_desired', 10)

        self.create_timer(Ts, self._control_loop)

        self.get_logger().info(
            f'MPCIBVSNode ready — N={N}, Ts={Ts}s, '
            f'v=[{v_min:.2f},{v_max:.2f}], omega_max={omega_max:.2f}, '
            f'OSQP={_OSQP_AVAILABLE}'
        )
        self.get_logger().info(f'Interaction matrix L_s*:\n{np.array2string(L_star, precision=4)}')

    def _feat_callback(self, msg: Float64MultiArray):
        self._feat = msg.data
        self._last_feat_time = self.get_clock().now()

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

        if conf < 0.5:
            self._publish_stop()
            return

        s0 = np.array([eu, ev, ea])
        t0 = time.monotonic()
        u_opt = self._mpc.solve(s0)
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
