"""t -> r mapping: the massless static potential in D dimensions.

The modular correlation G(t) ~ 1/t (verified in exp_gravity_modular_observer)
is the TIME-domain power law of a massless (scale-invariant) field.  The
SPATIAL correlation is the static potential V(r) = int d^D k e^{i k r} / k^2:

    D=1: V(r) ~ -|r|         (linear)
    D=2: V(r) ~ -(1/2 pi) log r   (log)
    D=3: V(r) = 1/(4 pi r)    (Coulomb 1/r)   <-- the one GR needs

This nails 卡点3 (1/r is 3D) AND 卡点2 (the light-cone r=t mapping, because
scale-invariance has no horizon scale, so t->r is linear not log).

Code: `py -m experiments.exp_gravity_tr_mapping`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import integrate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def static_potential_3d(r):
    """V(r) = int d^3k e^{ik.r}/k^2 = 1/(4 pi r).  Compute via angular integral."""
    # int_0^inf dk k^2 * (sin(kr)/(kr)) / k^2 * 4 pi = (4 pi / r) int_0^inf dk sin(kr)/k
    # = (4 pi / r) * (pi/2) = 2 pi^2 / r   [without (2pi)^3 normalization]
    # with (2pi)^3 normalization -> 1/(4 pi r)
    return 1.0 / (4.0 * np.pi * r)


def static_potential_numeric(r, D=3, kmax=100.0, n=20000):
    """Numerically compute V(r) = int d^D k e^{ikr}/k^2 (radial part)."""
    k = np.linspace(1e-6, kmax, n)
    # radial integrand: k^{D-1} * (D-dim angular factor) * (sin(kr)/(kr))^{...}
    # For D=3: 4 pi k^2 * sin(kr)/(kr) / k^2 = 4 pi sin(kr)/(k r)
    # For D=2: 2 pi k * J0(kr) / k^2 = 2 pi J0(kr)/k
    # For D=1: 2 * cos(kr) / k^2
    if D == 3:
        integrand = 4.0 * np.pi * np.sin(k * r) / (k * r) / (k ** 2) * (k ** 2)
        # = 4 pi sin(kr)/(k r)   [the k^2 from Jacobian cancels the 1/k^2]
        integrand = 4.0 * np.pi * np.sin(k * r) / (k * r)
        V = integrate.trapezoid(integrand, k)
    elif D == 2:
        from scipy.special import j0
        integrand = 2.0 * np.pi * k * j0(k * r) / (k ** 2)
        V = integrate.trapezoid(integrand, k)
    elif D == 1:
        integrand = 2.0 * np.cos(k * r) / (k ** 2)
        V = integrate.trapezoid(integrand, k)
    return V


def main():
    print("=" * 74)
    print("t -> r MAPPING: massless static potential in D dimensions")
    print("=" * 74)

    # 3D Coulomb: verify V(r) = 1/(4 pi r)
    print("\n--- 3D: V(r) = int d^3k e^{ikr}/k^2 = 1/(4 pi r) ---")
    print(f"  {'r':>5} {'numeric':>12} {'1/(4 pi r)':>12} {'ratio':>8}")
    for r in [0.5, 1.0, 2.0, 4.0, 8.0]:
        Vnum = static_potential_numeric(r, D=3)
        Vexact = static_potential_3d(r) * (2.0 * np.pi ** 2)  # undo (2pi)^3 to match numeric convention
        # numeric used no (2pi)^3, so compare to 2 pi^2 / r
        Vexact = 2.0 * np.pi ** 2 / r
        print(f"  {r:>5.1f} {Vnum:>12.5f} {Vexact:>12.5f} {Vnum/Vexact:>8.4f}")

    # dimensional ladder
    print("\n--- dimensional ladder (fit V(r) ~ r^{-p} or log) ---")
    print("  D=1: V(r) ~ -|r| (linear)   D=2: V(r) ~ -(1/2pi) log r   D=3: 1/r")
    # 3D fit
    rs = np.array([1.0, 2.0, 4.0, 8.0])
    V3 = np.array([static_potential_numeric(r, D=3) for r in rs])
    p3 = -np.polyfit(np.log(rs), np.log(V3), 1)[0]
    print(f"  3D fit: V ~ r^{{-{p3:.3f}}}  (expect 1.0)")

    # 2D fit (log)
    V2 = np.array([static_potential_numeric(r, D=2) for r in rs])
    # V2 should be ~ -log r, i.e., linear in log r with negative slope
    p2 = np.polyfit(np.log(rs), V2, 1)[0]
    print(f"  2D: V vs log r slope = {p2:.4f}  (expect ~ -1/(2pi) = -0.159)")

    # 1D fit (linear)
    V1 = np.array([static_potential_numeric(r, D=1) for r in rs])
    p1 = np.polyfit(rs, V1, 1)[0]
    print(f"  1D: V vs r slope = {p1:.4f}  (expect ~ -pi = -3.14, linear)")

    out = ROOT / "experiments" / "exp_gravity_tr_mapping_last_run.json"
    out.write_text(json.dumps({"p3": float(p3), "p2_slope": float(p2), "p1_slope": float(p1)},
                              indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
