"""Re-derive step 1b: WHY is delta blind to a pure x-direction modulus wave?

Found: r_x = 1+eps*cos(kx), r_y = 1  gives delta = 0 (to 1e-15).  This means the
angle defect is INSENSITIVE to an isotropic-rescaling-like perturbation along one
axis... OR it means delta only responds to ANISOTROPIC (shear) moduli, i.e. when
r_x and r_y differ in a way that changes the corner ANGLES.

Key insight to test: a rectangular lattice with r_x(x) and r_y=1 still has all
plaquettes as RECTANGLES (90-degree corners), so the angle defect is ZERO — the
corners stay right angles even though the lattice is "stretched" along x.  Only
when r_x AND r_y both vary in a way that makes corners non-right-angles does
delta become nonzero.

So the true curvature source is NOT "modulus Laplacian" but "corner-angle
deviation", which is second-order in the SHEAR (difference between r_x and r_y
gradients), not in the individual moduli.

Test:
  1. r_x = 1+eps*cos(kx), r_y = 1+eps*cos(kx)  (isotropic along x) -> expect 0
  2. r_x = 1+eps*cos(kx), r_y = 1+eps*cos(ky)  (shear)           -> expect nonzero?
  3. r_x = 1+eps*cos(kx), r_y = 1-eps*cos(kx)  (pure shear)      -> expect nonzero

Code: `py -m experiments.exp_bridge_B_rederive1b`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from itertools import product

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def angle_defect_field(r_x, r_y, L):
    delta = np.zeros((L, L))

    def wrap(idx):
        return idx % L

    for p, q in product(range(L), repeat=2):
        a = r_x[p, q]
        b = r_y[wrap(p + 1), q]
        c = r_x[p, wrap(q + 1)]
        d = r_y[p, q]
        diag = np.sqrt(a**2 + b**2)

        def tri_angles(x, y, z):
            ax = np.arccos(np.clip((y**2 + z**2 - x**2) / (2 * y * z), -1, 1))
            ay = np.arccos(np.clip((x**2 + z**2 - y**2) / (2 * x * z), -1, 1))
            az = np.arccos(np.clip((x**2 + y**2 - z**2) / (2 * x * y), -1, 1))
            return ax, ay, az

        angC, angA1, angB = tri_angles(a, b, diag)
        angD, angA2, angC2 = tri_angles(diag, c, d)
        angA = angA1 + angA2
        angC_tot = angC + angC2

        delta[p, q] += angA
        delta[wrap(p + 1), q] += angB
        delta[wrap(p + 1), wrap(q + 1)] += angC_tot
        delta[p, wrap(q + 1)] += angD

    return 2 * np.pi - delta


def main():
    L = 48
    x = np.arange(L); y = np.arange(L)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    eps = 0.1
    k = 2 * np.pi * 3 / L

    cases = {
        "isotropic x: rx=ry=1+eps cos(kx)": (1 + eps*np.cos(k*xx), 1 + eps*np.cos(k*xx)),
        "cross:      rx=1+eps cos(kx), ry=1+eps cos(ky)": (1 + eps*np.cos(k*xx), 1 + eps*np.cos(k*yy)),
        "pure shear: rx=1+eps cos(kx), ry=1-eps cos(kx)": (1 + eps*np.cos(k*xx), 1 - eps*np.cos(k*xx)),
    }
    print("=== re-derive 1b: which modulus perturbation actually curves? ===")
    print(f"  L={L}, eps={eps}, k = 3*(2pi/L)")
    for name, (rx, ry) in cases.items():
        delta = angle_defect_field(rx, ry, L)
        print(f"  {name:44s}: max|delta| = {np.max(np.abs(delta)):.3e}")

    # Also: does the angle defect only see the SHEAR (rx - ry)?
    # Build a shear field s(x) and check delta responds.
    print("\n  => reading: if only 'cross'/'pure shear' are nonzero, the curvature")
    print("     source is the SHEAR (anisotropy r_x vs r_y), not the modulus Laplacian.")

    summary = {
        "L": L, "eps": eps, "k_mode": 3,
        "isotropic_x_maxdelta": float(np.max(np.abs(angle_defect_field(
            1 + eps*np.cos(k*xx), 1 + eps*np.cos(k*xx), L)))),
        "cross_maxdelta": float(np.max(np.abs(angle_defect_field(
            1 + eps*np.cos(k*xx), 1 + eps*np.cos(k*yy), L)))),
        "shear_maxdelta": float(np.max(np.abs(angle_defect_field(
            1 + eps*np.cos(k*xx), 1 - eps*np.cos(k*xx), L)))),
    }
    out = ROOT / "experiments" / "exp_bridge_B_rederive1b_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
