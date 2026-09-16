"""Does 无外部观察者 (no external observer) force 实化 (realification)?

Conjecture: 自反性 = 无外部复相位 -> 实化.

Check: 无偏好方向 (a face of 无外部观察者) -> 结构选择门 -> pi-flux -> D REAL?
The pi-flux D has phases theta_ij in {0, pi} (holonomy -1), so D_ij = +- r is REAL.
So "无外部观察者 -> D 实" IS in the theory, via the structure-selection gate.

But: this realifies the SPATIAL part (modulus/phase), NOT the spin (Kramers T^2=-1).
So it's the FIRST step of 实化, not the full climb to R^8 (octonions).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def build_pi_flux(L):
    """LxL torus, pi flux per plaquette, real gauge (from exp_z2_interaction)."""
    n = L * L
    D = np.zeros((n, n))
    for x in range(L):
        for y in range(L):
            i = x + L * y
            D[i, (x + 1) % L + L * y] += 1.0
            D[i, x + L * ((y + 1) % L)] += 1.0 if x % 2 == 0 else -1.0
    return (D + D.T) / 2.0


def main() -> None:
    D = build_pi_flux(4)
    # phases: D_ij = |D_ij| e^{i theta_ij}; is theta in {0, pi}?  i.e. is D real?
    im_norm = np.max(np.abs(D.imag))
    phases = np.angle(D[D != 0]) / np.pi  # in units of pi
    phase_set = sorted(set(np.round(phases, 6).tolist()))

    print("pi-flux D (from structure-selection gate, L=4):")
    print(f"  max |Im D| = {im_norm:.2e}  -> D is REAL? {im_norm < 1e-12}")
    print(f"  phases theta/pi in {phase_set}  -> phases are {{0, pi}} (Z2, not continuous U(1))")
    print()
    print("Chain (already in the theory):")
    print("  无偏好方向 (无外部观察者) -> 模长相等 (定理1) -> 结构选择门 -> pi-flux -> D 实")
    print("  => 无外部观察者 DOES force D real (theta in {0,pi}) = 实化第一步 (spatial).")
    print()
    print("But NOT the full climb:")
    print("  - the SPIN part (Kramers T^2=-1) is still complex/quaternionic;")
    print("  - full 实化 (-> R^8 octonions) needs the spin part too;")
    print("  - so 无外部观察者 forces the SPATIAL realification, but the spin->octonion")
    print("    climb (唯一性) is still not forced by this chain.")

    out = ROOT / "experiments" / "exp_no_external_observer_realification_last_run.json"
    out.write_text(json.dumps({
        "D_real": bool(im_norm < 1e-12),
        "phase_set": phase_set,
        "max_im": float(im_norm),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
