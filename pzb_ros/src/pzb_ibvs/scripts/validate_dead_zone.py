#!/usr/bin/env python3
"""
Validate dead-zone hysteresis fix.

Simulates a feature error that drifts slowly in and out of the dead-zone
boundary. Shows that the old step-function dead zone causes chattering while
the new hysteresis version gives a smooth single transition.

Usage:
    python3 src/pzb_ibvs/scripts/validate_dead_zone.py
"""

import numpy as np
import matplotlib.pyplot as plt


def simulate_dead_zone(errors, dz_enter, dz_exit_factor, use_hysteresis):
    """
    errors: array of eu values (1D scenario, ev/ea treated as 0)
    Returns: array of servo booleans and slew-limited v_cmd.
    """
    dz_exit = dz_enter * dz_exit_factor
    in_dz = False
    Ts = 0.08
    slew = 0.20  # m/s²
    v_max = 0.25

    servos, vcmds = [], []
    v_prev = 0.0

    for eu in errors:
        if use_hysteresis:
            if in_dz:
                if abs(eu) > dz_exit:
                    in_dz = False
            else:
                if abs(eu) < dz_enter:
                    in_dz = True
            servo = not in_dz
        else:
            servo = abs(eu) >= dz_enter

        target_v = v_max if servo else 0.0
        dv = slew * Ts
        v = v_prev + np.clip(target_v - v_prev, -dv, dv)
        v_prev = v
        servos.append(servo)
        vcmds.append(v)

    return np.array(servos, dtype=float), np.array(vcmds)


def main():
    dz = 20.0          # entry threshold (pixels)
    exit_factor = 1.5  # exit at dz * 1.5 = 30 px

    # Craft an error signal that oscillates near the boundary:
    # 0→50px (approach goal) then add ±3px noise around 18px (inside/near boundary)
    t = np.linspace(0, 30, 375)   # 30 s at 12.5 Hz
    eu = np.zeros_like(t)
    # Phase 1: far away, approaching
    eu[:100] = 80 - 0.6 * np.arange(100)          # 80→20 px
    # Phase 2: near boundary with noise (chattering zone)
    noise = 5.0 * np.sin(2 * np.pi * 1.5 * t[100:250])  # ±5 px oscillation around 18 px
    eu[100:250] = 18 + noise
    # Phase 3: disturbance pushes far out, then back
    eu[250:300] = np.linspace(18, 60, 50)
    eu[300:350] = np.linspace(60, 18, 50)
    eu[350:] = 18 + noise[:25]

    s_old, v_old = simulate_dead_zone(eu, dz, exit_factor, use_hysteresis=False)
    s_new, v_new = simulate_dead_zone(eu, dz, exit_factor, use_hysteresis=True)

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(t, eu, 'k', lw=1)
    axes[0].axhline(dz,              color='r',   ls='--', lw=1, label=f'enter threshold ({dz:.0f} px)')
    axes[0].axhline(dz * exit_factor, color='orange', ls='--', lw=1,
                    label=f'exit threshold ({dz*exit_factor:.0f} px)')
    axes[0].set_ylabel('eu  [px]')
    axes[0].set_title('Feature error eu')
    axes[0].legend(fontsize=8)

    axes[1].plot(t, v_old, 'r', lw=1.2, label='old (step)')
    axes[1].plot(t, v_new, 'g', lw=1.2, label='new (hysteresis)')
    axes[1].axhline(0, color='grey', ls='--', lw=0.6)
    axes[1].set_ylabel('v_cmd  [m/s]')
    axes[1].set_title('Linear velocity command — dead-zone comparison')
    axes[1].legend(fontsize=8)

    # Count transitions (chatter metric)
    old_trans = int(np.sum(np.abs(np.diff(s_old)) > 0.5))
    new_trans = int(np.sum(np.abs(np.diff(s_new)) > 0.5))
    axes[2].plot(t, s_old, 'r', lw=1.2, alpha=0.7, label=f'old — {old_trans} transitions')
    axes[2].plot(t, s_new, 'g', lw=1.2, alpha=0.7, label=f'new — {new_trans} transitions')
    axes[2].set_ylabel('servo active')
    axes[2].set_xlabel('Time  [s]')
    axes[2].set_title('Servo enable flag')
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    out = 'dead_zone_validation.png'
    plt.savefig(out, dpi=120)
    print(f'Saved → {out}')
    print(f'Dead-zone transitions: old={old_trans}, new={new_trans}')
    plt.show()


if __name__ == '__main__':
    main()
