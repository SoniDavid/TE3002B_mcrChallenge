#!/usr/bin/env python3
"""
Closed-loop simulation of the MPC IBVS controller plus the production
downstream chain (twist_slew_limiter → velocity_controller PI → MCU velocity
loop), using the *real* MPC classes imported from mpc_ibvs_node.py.

Scenarios (stacked into one figure, pzb_ros/mpc_simulation.png):

  Row 1  Nominal convergence — feature errors vs time, and the velocity as it
         passes MPC → slew → PI → robot. eu and ea converge to 0; ev parks at a
         non-zero value because it is intentionally un-penalized (Q_diag[1]=0):
         once eu≈0 the vertical-pixel error and the area error are driven
         essentially only by forward v (a 1-DOF redundancy), so distance is
         regulated via the area channel alone.
  Row 2  Velocity detail (zoom) + angular velocity — confirms the raw MPC
         command is smooth (no bang-bang) after the feature-error normalization.
  Row 3  Feature loss (slew limiter BYPASSED) — soft-stop ramp vs the old
         step-to-zero. With the production slew limiter active the two are
         indistinguishable; this panel is the only place the soft-stop's own
         falling edge is visible.
  Row 4  PI rising edge — velocity_controller alone, clamped vs unclamped PI
         correction, confirming the recovery-edge overshoot drop (≈50% → 30%).

Usage:
    python3 src/pzb_ibvs/scripts/simulate_mpc_closed_loop.py
"""

import os
import sys
import importlib.util
import types

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Import the real MPC helpers without dragging in rclpy ────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_NODE_PATH = os.path.join(_THIS_DIR, 'mpc_ibvs_node.py')
_REPO_PZB = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))  # .../pzb_ros


def _stub_ros_modules():
    for name in ('rclpy', 'rclpy.node', 'rclpy.qos',
                 'geometry_msgs', 'geometry_msgs.msg',
                 'std_msgs', 'std_msgs.msg'):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules['rclpy'].init = lambda *a, **k: None
    sys.modules['rclpy'].spin = lambda *a, **k: None
    sys.modules['rclpy'].try_shutdown = lambda *a, **k: None
    sys.modules['rclpy.node'].Node = type('Node', (), {})

    class _QoS:
        def __init__(self, **kw):
            pass
    sys.modules['rclpy.qos'].QoSProfile = _QoS
    sys.modules['rclpy.qos'].ReliabilityPolicy = type('R', (), {'BEST_EFFORT': 0})
    sys.modules['rclpy.qos'].DurabilityPolicy = type('D', (), {'VOLATILE': 0})
    sys.modules['rclpy.qos'].HistoryPolicy = type('H', (), {'KEEP_LAST': 0})
    sys.modules['geometry_msgs.msg'].Twist = type('Twist', (), {})
    sys.modules['geometry_msgs.msg'].TwistStamped = type('TwistStamped', (), {})
    sys.modules['std_msgs.msg'].Float64MultiArray = type('Float64MultiArray', (), {})


_stub_ros_modules()
_spec = importlib.util.spec_from_file_location('mpc_ibvs_node', _NODE_PATH)
mpc_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mpc_mod)

LinearMPC = mpc_mod.LinearMPC
build_L = mpc_mod._build_interaction_matrix
build_AB = mpc_mod._build_augmented_model


# ── Parameters (mirror config/mpc_ibvs_params.yaml + pid_vel_params.yaml) ────
P = dict(
    Ts=0.08, N=15,
    fx=486.2936, fy=488.4081, Z_star=0.50, desired_area=2862.0,
    tau_v=0.40, tau_omega=0.40,
    v_min=-0.10, v_max=0.25, omega_max=0.45,
    Q_diag=[1.0, 0.0, 0.8],            # ev dropped — 1-DOF redundancy with ea
    R_diag=[0.40, 1.00],
    feat_norm_uv=600.0, feat_norm_a=50.0,   # Q applied to normalized errors
    lost_decel_s=0.40,
    # downstream
    slew_lin_a=0.20, slew_ang_a=0.50,
    pi_kp_v=0.5, pi_ki_v=0.1, pi_kp_w=0.5, pi_ki_w=0.0, pi_ic=0.2,
    pi_clip_frac=0.3, pi_clip_floor_v=0.05, pi_clip_floor_w=0.15,
    pi_max_v=0.40, pi_max_w=1.50,
    tau_mcu=0.06,   # fast MCU velocity loop (Ti=0.05 on firmware)
)


def make_mpc():
    L0 = build_L(P['fx'], P['fy'], P['Z_star'], P['desired_area'])
    A, B = build_AB(L0, P['Ts'], P['tau_v'], P['tau_omega'])
    # Mirror mpc_ibvs_node: Q applied to normalized feature errors.
    fn_sq = [P['feat_norm_uv'] ** 2, P['feat_norm_uv'] ** 2, P['feat_norm_a'] ** 2]
    Q_diag_full = [q / n for q, n in zip(P['Q_diag'], fn_sq)] + [0.0, 0.0]
    Q = np.diag(Q_diag_full)
    R = np.diag(P['R_diag'])
    u_min = np.array([P['v_min'], -P['omega_max']])
    u_max = np.array([P['v_max'],  P['omega_max']])
    return LinearMPC(A, B, Q, R, P['N'], u_min, u_max)


# ── One closed-loop run ──────────────────────────────────────────────────────
def run(T, eu0, ev0, ea0, lost_window=None, use_soft_stop=True, use_slew=True,
        use_pi=True):
    """
    Pipeline: MPC → [slew limiter] → [PI inner] → first-order MCU loop → plant.
    Image features integrate with the *actual* robot velocity.
    `lost_window` = (t0, t1) during which features are reported stale.
    Returns a dict of equal-length time series.
    """
    Ts = P['Ts']
    mpc = make_mpc()
    eu, ev, ea = float(eu0), float(ev0), float(ea0)
    v_act = w_act = 0.0
    v_slew = w_slew = 0.0
    iv = iw = 0.0
    last_v_mpc = last_w_mpc = 0.0

    n = int(T / Ts)
    t = np.arange(n) * Ts
    out = {k: np.zeros(n) for k in
           ('eu', 'ev', 'ea', 'v_mpc', 'w_mpc', 'v_slew', 'w_slew',
            'v_cmd', 'w_cmd', 'v_act', 'w_act', 'lost')}

    for k in range(n):
        out['eu'][k], out['ev'][k], out['ea'][k] = eu, ev, ea
        is_lost = lost_window is not None and lost_window[0] <= t[k] < lost_window[1]
        out['lost'][k] = float(is_lost)

        # Local interaction matrix (LTV refresh, like the node does)
        L = build_L(P['fx'], P['fy'], P['Z_star'], P['desired_area'],
                    eu=eu, ev=ev)

        if not is_lost:
            A, B = build_AB(L, Ts, P['tau_v'], P['tau_omega'])
            mpc.update_model(A, B)
            u = mpc.solve(np.array([eu, ev, ea, v_act, w_act]))
            v_mpc = float(np.clip(u[0], P['v_min'], P['v_max']))
            w_mpc = float(np.clip(u[1], -P['omega_max'], P['omega_max']))
        else:
            # Soft-stop ramp from mpc_ibvs_node._control_loop (else branch).
            if use_soft_stop and P['lost_decel_s'] > 0.0:
                decay = Ts / P['lost_decel_s']
                v_mpc = last_v_mpc * max(0.0, 1.0 - decay)
                w_mpc = last_w_mpc * max(0.0, 1.0 - decay)
                if abs(v_mpc) < 1e-3:
                    v_mpc = 0.0
                if abs(w_mpc) < 1e-3:
                    w_mpc = 0.0
            else:
                v_mpc = w_mpc = 0.0

        out['v_mpc'][k], out['w_mpc'][k] = v_mpc, w_mpc
        last_v_mpc, last_w_mpc = v_mpc, w_mpc

        # Slew limiter
        if use_slew:
            v_slew += np.clip(v_mpc - v_slew, -P['slew_lin_a'] * Ts,  P['slew_lin_a'] * Ts)
            w_slew += np.clip(w_mpc - w_slew, -P['slew_ang_a'] * Ts,  P['slew_ang_a'] * Ts)
        else:
            v_slew, w_slew = v_mpc, w_mpc
        out['v_slew'][k], out['w_slew'][k] = v_slew, w_slew

        # PI inner loop (velocity_controller), with the new correction clamp
        if use_pi:
            iv = float(np.clip(iv + (v_slew - v_act) * Ts, -P['pi_ic'], P['pi_ic']))
            iw = float(np.clip(iw + (w_slew - w_act) * Ts, -P['pi_ic'], P['pi_ic']))
            raw_cv = P['pi_kp_v'] * (v_slew - v_act) + P['pi_ki_v'] * iv
            raw_cw = P['pi_kp_w'] * (w_slew - w_act) + P['pi_ki_w'] * iw
            cv = float(np.clip(raw_cv, -P['pi_clip_frac'] * max(abs(v_slew), P['pi_clip_floor_v']),
                                        P['pi_clip_frac'] * max(abs(v_slew), P['pi_clip_floor_v'])))
            cw = float(np.clip(raw_cw, -P['pi_clip_frac'] * max(abs(w_slew), P['pi_clip_floor_w']),
                                        P['pi_clip_frac'] * max(abs(w_slew), P['pi_clip_floor_w'])))
            v_cmd = float(np.clip(v_slew + cv, -P['pi_max_v'], P['pi_max_v']))
            w_cmd = float(np.clip(w_slew + cw, -P['pi_max_w'], P['pi_max_w']))
        else:
            v_cmd, w_cmd = v_slew, w_slew
        out['v_cmd'][k], out['w_cmd'][k] = v_cmd, w_cmd

        # MCU velocity loop — fast first order, exact ZOH (tau_mcu may be < Ts,
        # so a forward-Euler step would be unstable/ringy — use exp form).
        a = 1.0 - np.exp(-Ts / P['tau_mcu'])
        v_act = (1 - a) * v_act + a * v_cmd
        w_act = (1 - a) * w_act + a * w_cmd
        out['v_act'][k], out['w_act'][k] = v_act, w_act

        # Image kinematics driven by ACTUAL velocity (Euler, matches node model)
        eu += Ts * (L[0, 0] * v_act + L[0, 1] * w_act)
        ev += Ts * (L[1, 0] * v_act + L[1, 1] * w_act)
        ea += Ts * (L[2, 0] * v_act)

    out['t'] = t
    return out


# ── PI-only rising-edge test ────────────────────────────────────────────────
def run_pi_step(T, v_des_target, clip_enabled):
    Ts = 0.02            # PI runs at 50 Hz
    n = int(T / Ts)
    t = np.arange(n) * Ts
    v_des = np.where(t >= 0.20, v_des_target, 0.0)
    v_act = 0.0
    iv = 0.0
    v_cmd_h = np.zeros(n)
    v_act_h = np.zeros(n)
    corr_h = np.zeros(n)
    tau = 0.40           # mimics the slew-limited cascade response
    for k in range(n):
        iv = float(np.clip(iv + (v_des[k] - v_act) * Ts, -P['pi_ic'], P['pi_ic']))
        raw = P['pi_kp_v'] * (v_des[k] - v_act) + P['pi_ki_v'] * iv
        if clip_enabled:
            c = P['pi_clip_frac'] * max(abs(v_des[k]), P['pi_clip_floor_v'])
            corr = float(np.clip(raw, -c, c))
        else:
            corr = raw
        v_cmd = float(np.clip(v_des[k] + corr, -P['pi_max_v'], P['pi_max_v']))
        v_cmd_h[k] = v_cmd
        corr_h[k] = corr
        a = 1.0 - np.exp(-Ts / tau)
        v_act = (1 - a) * v_act + a * v_cmd
        v_act_h[k] = v_act
    return dict(t=t, v_des=v_des, v_cmd=v_cmd_h, v_act=v_act_h, corr=corr_h)


# ── Plotting helpers ────────────────────────────────────────────────────────
def _shade_lost(ax, t, lost):
    if not np.any(lost > 0):
        return
    inb = False
    start = 0.0
    for i in range(len(t)):
        if lost[i] > 0 and not inb:
            start = t[i]; inb = True
        elif inb and (lost[i] == 0 or i == len(t) - 1):
            ax.axvspan(start, t[i], color='red', alpha=0.12, zorder=0)
            inb = False


def main():
    # Physically consistent feature errors: for a ground target with reference
    # row cy=360, "far" → ev<0 AND ea<0; "near" → both >0. eu sets heading.
    ic_far  = (-160.0, -55.0, -22.0)   # target off to one side, a bit far

    print('[1/4] Nominal convergence (full chain)...')
    nom = run(18.0, *ic_far)

    print('[2/4] Same run — raw vs slew-limited command...')  # reuse nom

    print('[3/4] Feature loss, slew limiter BYPASSED (soft-stop vs step)...')
    soft = run(12.0, *ic_far, lost_window=(4.0, 6.0), use_soft_stop=True,  use_slew=False)
    hard = run(12.0, *ic_far, lost_window=(4.0, 6.0), use_soft_stop=False, use_slew=False)

    print('[4/4] PI rising edge (clamp vs no clamp)...')
    pic = run_pi_step(T=2.0, v_des_target=0.20, clip_enabled=True)
    pin = run_pi_step(T=2.0, v_des_target=0.20, clip_enabled=False)

    fig, ax = plt.subplots(4, 2, figsize=(14, 13))
    fig.suptitle('MPC IBVS — closed-loop simulation (real MPC code + slew + PI + MCU)',
                 fontsize=13, y=0.997)

    # Row 1 — nominal
    a = ax[0, 0]
    a.plot(nom['t'], nom['eu'], label='eu (px)')
    a.plot(nom['t'], nom['ev'], label='ev (px)')
    a.plot(nom['t'], nom['ea'], label='ea (px)')
    for thr, c in [(20, 'C0'), (-20, 'C0'), (20, 'C1'), (-20, 'C1')]:
        pass
    a.axhline(0, color='k', lw=0.6, ls='--')
    a.set_title('Nominal: feature errors  (eu, ea → 0;  ev un-penalized, parks ≠ 0)')
    a.set_ylabel('pixels'); a.legend(loc='upper right'); a.grid(alpha=0.3)
    a.set_xlabel('time (s)')

    a = ax[0, 1]
    a.plot(nom['t'], nom['v_mpc'],  label='v_mpc (raw)', alpha=0.45)
    a.plot(nom['t'], nom['v_slew'], label='v_slew', alpha=0.8)
    a.plot(nom['t'], nom['v_cmd'],  label='v_cmd→MCU', alpha=0.8)
    a.plot(nom['t'], nom['v_act'],  label='v_actual', lw=2.2)
    a.axhline(P['v_max'], color='r', lw=0.6, ls=':'); a.axhline(P['v_min'], color='r', lw=0.6, ls=':')
    a.set_title('Nominal: linear velocity through the chain')
    a.set_ylabel('m/s'); a.legend(loc='lower right'); a.grid(alpha=0.3)
    a.set_xlabel('time (s)')

    # Row 2 — raw vs slew detail, plus omega
    a = ax[1, 0]
    a.plot(nom['t'], nom['v_mpc'],  label='v_mpc (raw)', lw=1.0)
    a.plot(nom['t'], nom['v_act'],  label='v_actual', lw=2.2)
    a.set_xlim(8, 14)
    a.set_title('Nominal (zoom 8–14 s): raw MPC command is smooth\nafter feature-error normalization (no bang-bang)')
    a.set_ylabel('m/s'); a.legend(loc='upper right'); a.grid(alpha=0.3)
    a.set_xlabel('time (s)')

    a = ax[1, 1]
    a.plot(nom['t'], nom['w_mpc'],  label='ω_mpc (raw)', alpha=0.45)
    a.plot(nom['t'], nom['w_slew'], label='ω_slew', alpha=0.8)
    a.plot(nom['t'], nom['w_cmd'],  label='ω_cmd→MCU', alpha=0.8)
    a.plot(nom['t'], nom['w_act'],  label='ω_actual', lw=2.2)
    a.axhline(P['omega_max'], color='r', lw=0.6, ls=':'); a.axhline(-P['omega_max'], color='r', lw=0.6, ls=':')
    a.set_title('Nominal: angular velocity through the chain')
    a.set_ylabel('rad/s'); a.legend(loc='upper right'); a.grid(alpha=0.3)
    a.set_xlabel('time (s)')

    # Row 3 — feature loss, no slew
    a = ax[2, 0]
    a.plot(soft['t'], soft['v_mpc'], label='soft-stop v_mpc')
    a.plot(hard['t'], hard['v_mpc'], label='step v_mpc', ls='--')
    _shade_lost(a, soft['t'], soft['lost'])
    a.set_title('Feature loss (slew bypassed): MPC linear output — falling edge')
    a.set_ylabel('m/s'); a.legend(loc='upper right'); a.grid(alpha=0.3)
    a.set_xlabel('time (s)')

    a = ax[2, 1]
    a.plot(soft['t'], soft['v_act'], label='soft-stop v_actual')
    a.plot(hard['t'], hard['v_act'], label='step v_actual', ls='--')
    _shade_lost(a, soft['t'], soft['lost'])
    a.set_title('Feature loss (slew bypassed): robot actual velocity')
    a.set_ylabel('m/s'); a.legend(loc='upper right'); a.grid(alpha=0.3)
    a.set_xlabel('time (s)')

    # Row 4 — PI rising edge
    a = ax[3, 0]
    a.plot(pin['t'], pin['v_cmd'], label='no clamp v_cmd', ls='--')
    a.plot(pic['t'], pic['v_cmd'], label='clamped v_cmd')
    a.plot(pic['t'], pic['v_des'], label='v_des', color='k', lw=1, alpha=0.6)
    a.axhline(0.20 * 1.25, color='r', lw=0.6, ls=':', label='+25%')
    a.set_title('PI rising edge: v_cmd vs v_des  (overshoot ≈50% → 30%)')
    a.set_ylabel('m/s'); a.legend(loc='lower right'); a.grid(alpha=0.3)
    a.set_xlabel('time (s)')

    a = ax[3, 1]
    a.plot(pin['t'], pin['corr'], label='no clamp PI corr', ls='--')
    a.plot(pic['t'], pic['corr'], label='clamped PI corr')
    a.set_title('PI rising edge: feedback correction term')
    a.set_ylabel('m/s'); a.legend(loc='upper right'); a.grid(alpha=0.3)
    a.set_xlabel('time (s)')

    plt.tight_layout()
    out_path = os.path.join(_REPO_PZB, 'mpc_simulation.png')
    plt.savefig(out_path, dpi=125, bbox_inches='tight')
    print(f'\nSaved: {out_path}')

    # ── Numeric summary ────────────────────────────────────────────────────
    def settle(t, x, tol):
        idx = np.where(np.abs(x) > tol)[0]
        return float(t[idx[-1]]) if len(idx) else 0.0

    tail = slice(int(12.0 / P['Ts']), None)   # last ~6 s of nominal
    print('\n=== Summary (nominal) ===')
    print(f'  eu: final={nom["eu"][-1]:+.1f} px   settle(|eu|>15px)={settle(nom["t"], nom["eu"], 15):.1f} s')
    print(f'  ea: final={nom["ea"][-1]:+.1f} px   settle(|ea|>15px)={settle(nom["t"], nom["ea"], 15):.1f} s')
    print(f'  ev: final={nom["ev"][-1]:+.1f} px   (un-penalized — parks ≠ 0 by design)')
    print(f'  v_mpc (raw)  tail std = {nom["v_mpc"][tail].std():.3f} m/s   range=[{nom["v_mpc"][tail].min():+.3f},{nom["v_mpc"][tail].max():+.3f}]')
    print(f'  v_actual     tail std = {nom["v_act"][tail].std():.3f} m/s   (slew + MCU loop attenuation)')
    print(f'  ω_actual     tail std = {nom["w_act"][tail].std():.3f} rad/s')
    print(f'  max |v_cmd| = {np.max(np.abs(nom["v_cmd"])):.3f} m/s    max |ω_cmd| = {np.max(np.abs(nom["w_cmd"])):.3f} rad/s')

    win = slice(int(3.9 / P['Ts']), int(4.3 / P['Ts']))
    print('\n=== Summary (feature loss, slew bypassed) ===')
    print(f'  soft-stop: max |Δv_mpc| at loss = {np.max(np.abs(np.diff(soft["v_mpc"][win]))):.4f} m/s/cycle')
    print(f'  step     : max |Δv_mpc| at loss = {np.max(np.abs(np.diff(hard["v_mpc"][win]))):.4f} m/s/cycle')

    print('\n=== Summary (PI rising edge) ===')
    oc = np.max(pic['v_cmd']) - 0.20
    on = np.max(pin['v_cmd']) - 0.20
    print(f'  peak overshoot WITH clamp    = {oc*1000:5.1f} mm/s ({oc/0.20*100:5.1f}%)')
    print(f'  peak overshoot WITHOUT clamp = {on*1000:5.1f} mm/s ({on/0.20*100:5.1f}%)')


if __name__ == '__main__':
    main()
