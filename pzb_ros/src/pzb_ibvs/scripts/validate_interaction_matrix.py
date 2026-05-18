#!/usr/bin/env python3
"""
Validate the interaction-matrix sign fix against the recorded rosbag.

Loads /visual_features and /cmd_vel_desired from mpc_behaviour_0.db3, then
re-runs the MPC on the same feature sequence with:
  - OLD model: L[2,0] = -sqrt(A_des)/Z  (as originally coded — wrong)
  - NEW model: L[2,0] = +sqrt(A_des)/Z  (corrected sign)

Plots v_cmd and omega_cmd for both, alongside ea, to make the correction
visually obvious.

Usage:
    cd /home/soni/Documents/classes/IRS_6to/manchesterRobotics/TE3002B_mcrChallenge/pzb_ros
    python3 src/pzb_ibvs/scripts/validate_interaction_matrix.py
"""

import sqlite3
import struct
import sys
import numpy as np
import matplotlib.pyplot as plt

# ── Inline MPC primitives (no ROS needed) ────────────────────────────────────

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def build_L(fx, fy, Z_star, A_des, eu, ev, sign=+1):
    sqrt_A = np.sqrt(A_des)
    xn = eu / fx if fx else 0.0
    yn = ev / fy if fy else 0.0
    return np.array([
        [eu / Z_star,           fx * (1.0 + xn * xn)],
        [ev / Z_star,           fy * (xn * yn)       ],
        [sign * sqrt_A / Z_star, 0.0                 ],
    ])


def build_AB(L, Ts, tau_v, tau_o):
    av = Ts / tau_v
    ao = Ts / tau_o
    A = np.array([
        [1., 0., 0., Ts*L[0,0], Ts*L[0,1]],
        [0., 1., 0., Ts*L[1,0], Ts*L[1,1]],
        [0., 0., 1., Ts*L[2,0], 0.        ],
        [0., 0., 0., 1.-av,     0.        ],
        [0., 0., 0., 0.,        1.-ao     ],
    ])
    B = np.array([
        [0., 0.], [0., 0.], [0., 0.],
        [av, 0.], [0., ao],
    ])
    return A, B


def solve_mpc_unconstrained(A, B, Q, R, N, x0, v_min, v_max, omega_max):
    nx, nu = B.shape
    S_x = np.zeros((N*nx, nx))
    S_u = np.zeros((N*nx, N*nu))
    Ak = np.eye(nx)
    for k in range(N):
        Ak = A @ Ak
        S_x[k*nx:(k+1)*nx] = Ak
        for j in range(k+1):
            S_u[k*nx:(k+1)*nx, j*nu:(j+1)*nu] = np.linalg.matrix_power(A, k-j) @ B
    Q_bar = np.kron(np.eye(N), Q)
    R_bar = np.kron(np.eye(N), R)
    H = S_u.T @ Q_bar @ S_u + R_bar
    H = (H + H.T) / 2
    f = S_u.T @ Q_bar @ (S_x @ x0)
    try:
        U = np.linalg.solve(H, -f)
    except np.linalg.LinAlgError:
        U = np.linalg.lstsq(H, -f, rcond=None)[0]
    lb = np.tile([v_min, -omega_max], N)
    ub = np.tile([v_max,  omega_max], N)
    U = np.clip(U, lb, ub)
    return U[:nu]


# ── Rosbag parsing ────────────────────────────────────────────────────────────

def parse_float64array(data):
    return struct.unpack_from('<4d', data, 20)  # eu, ev, ea, conf

def parse_twist(data):
    vals = struct.unpack_from('<6d', data, 4)
    return vals[0], vals[5]  # linear.x, angular.z


def load_bag(bag_path):
    conn = sqlite3.connect(bag_path)
    cur = conn.cursor()
    cur.execute('SELECT id, name FROM topics')
    topics = {row[1]: row[0] for row in cur.fetchall()}

    feat_id = topics.get('/visual_features')
    cmd_id  = topics.get('/cmd_vel_desired')
    if feat_id is None or cmd_id is None:
        print('Required topics not found in bag.', file=sys.stderr)
        sys.exit(1)

    cur.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp', (feat_id,))
    feat_rows = cur.fetchall()
    t0 = feat_rows[0][0] / 1e9

    feats, cmds_actual = [], []
    for ts, data in feat_rows:
        try:
            eu, ev, ea, conf = parse_float64array(data)
            feats.append((ts/1e9 - t0, eu, ev, ea, conf))
        except Exception:
            pass

    cur.execute('SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp', (cmd_id,))
    for ts, data in cur.fetchall():
        try:
            v, w = parse_twist(data)
            cmds_actual.append((ts/1e9 - t0, v, w))
        except Exception:
            pass

    conn.close()
    return feats, cmds_actual, t0


# ── Main ──────────────────────────────────────────────────────────────────────

def simulate(feats, sign, params):
    Ts       = params['Ts']
    N        = params['N']
    fx       = params['fx']
    fy       = params['fy']
    Z_star   = params['Z_star']
    A_des    = params['A_des']
    tau_v    = params['tau_v']
    tau_o    = params['tau_o']
    v_min    = params['v_min']
    v_max    = params['v_max']
    w_max    = params['omega_max']
    conf_thr = params['conf_thr']
    slew_v   = params['v_slew_rate']
    slew_w   = params['omega_slew_rate']

    Q = np.diag([1.0, 1.0, 0.8, 0.0, 0.0])
    R = np.diag([0.25, 0.80])

    v_prev = w_prev = 0.0
    times, vs, ws = [], [], []

    for t, eu, ev, ea, conf in feats:
        target_v = target_w = 0.0
        if conf >= conf_thr:
            L = build_L(fx, fy, Z_star, A_des, eu, ev, sign=sign)
            A, B = build_AB(L, Ts, tau_v, tau_o)
            x0 = np.array([eu, ev, ea, v_prev, w_prev])
            u = solve_mpc_unconstrained(A, B, Q, R, N, x0, v_min, v_max, w_max)
            target_v = _clamp(float(u[0]), v_min, v_max)
            target_w = _clamp(float(u[1]), -w_max, w_max)

        dv = slew_v * Ts
        dw = slew_w * Ts
        v = v_prev + _clamp(target_v - v_prev, -dv, dv)
        w = w_prev + _clamp(target_w - w_prev, -dw, dw)
        v_prev, w_prev = v, w

        times.append(t)
        vs.append(v)
        ws.append(w)

    return np.array(times), np.array(vs), np.array(ws)


def main():
    bag_path = 'mpc_behaviour/mpc_behaviour_0.db3'
    feats, cmds_actual, _ = load_bag(bag_path)

    params = dict(
        Ts=0.08, N=15, fx=486.2936, fy=488.4081,
        Z_star=0.5, A_des=2862.0, tau_v=0.15, tau_o=0.20,
        v_min=-0.10, v_max=0.25, omega_max=0.45,
        conf_thr=0.5, v_slew_rate=0.20, omega_slew_rate=0.60,
    )

    print('Simulating OLD model (sign = −1)…')
    t_old, v_old, w_old = simulate(feats, sign=-1, params=params)
    print('Simulating NEW model (sign = +1)…')
    t_new, v_new, w_new = simulate(feats, sign=+1, params=params)

    # Actual commands from bag
    t_act = np.array([r[0] for r in cmds_actual])
    v_act = np.array([r[1] for r in cmds_actual])
    w_act = np.array([r[2] for r in cmds_actual])

    # Feature ea timeline
    t_f  = np.array([r[0] for r in feats])
    ea_f = np.array([r[3] for r in feats])
    conf_f = np.array([r[4] for r in feats])

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(t_f, ea_f, 'k', lw=0.8, label='ea (from bag)')
    axes[0].axhline(0, color='grey', ls='--', lw=0.6)
    axes[0].set_ylabel('ea  [px]')
    axes[0].set_title('Area error (ea > 0 = too close, ea < 0 = too far)')
    axes[0].legend(fontsize=8)

    axes[1].plot(t_act, v_act, 'k',   lw=0.8, alpha=0.5, label='actual (bag)')
    axes[1].plot(t_old, v_old, 'r--', lw=1.2, label='old sign (−)')
    axes[1].plot(t_new, v_new, 'g',   lw=1.2, label='new sign (+)')
    axes[1].axhline(0, color='grey', ls='--', lw=0.6)
    axes[1].set_ylabel('v_cmd  [m/s]')
    axes[1].set_title('Linear velocity command — sign fix effect')
    axes[1].legend(fontsize=8)

    axes[2].plot(t_act, w_act, 'k',   lw=0.8, alpha=0.5, label='actual (bag)')
    axes[2].plot(t_old, w_old, 'r--', lw=1.2, label='old sign (−)')
    axes[2].plot(t_new, w_new, 'g',   lw=1.2, label='new sign (+)')
    axes[2].axhline(0, color='grey', ls='--', lw=0.6)
    axes[2].set_ylabel('ω_cmd  [rad/s]')
    axes[2].set_xlabel('Time  [s]')
    axes[2].set_title('Angular velocity command')
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    out = 'interaction_matrix_validation.png'
    plt.savefig(out, dpi=120)
    print(f'Saved → {out}')

    # Print summary statistics
    neg_old = np.sum(v_old < -0.005)
    neg_new = np.sum(v_new < -0.005)
    print(f'\nBackward commands (v < −0.005):')
    print(f'  Old model: {neg_old}/{len(v_old)} = {100*neg_old/len(v_old):.1f}%')
    print(f'  New model: {neg_new}/{len(v_new)} = {100*neg_new/len(v_new):.1f}%')
    plt.show()


if __name__ == '__main__':
    main()
