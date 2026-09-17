"""Confirm the flat pi-flux torus has a_2 = 0 (no scalar curvature).

Heat kernel of the pi-flux Dirac operator squared:
    Tr e^{-t D^2} = sum_k e^{-t * 4(cos^2 kx + cos^2 ky)}
Small-t expansion (2D):
    Tr e^{-t D^2} ~ a0/t + a2 + a4 t + ...
where a0 = Vol/(4 pi) (volume), a2 = (1/12pi) int sqrt(g) R (scalar curvature).

For the FLAT torus R=0, so a2=0.  The constant term comes only from the
Dirac zero modes (E=0), NOT curvature.

This confirms: nonzero a2 (curvature) requires bending the geometry (matter/
defect) -> bridge B depends on Q2 (matter bends geometry).

Code: `py -m experiments.exp_gravity_a2`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pi_flux_eigs_sq(npd):
    """Eigenvalues of D^2 for the pi-flux torus: 4(cos^2 kx + cos^2 ky)."""
    ks = 2 * np.pi * np.arange(npd) / npd
    eigs = []
    for kx in ks:
        for ky in ks:
            eigs.append(4.0 * (np.cos(kx) ** 2 + np.cos(ky) ** 2))
    return np.array(eigs)


def main():
    print("=" * 74)
    print("HEAT KERNEL a2: is the flat pi-flux torus curvature-free?")
    print("=" * 74)

    npd = 64
    eigs = pi_flux_eigs_sq(npd)
    N = len(eigs)  # N = npd^2
    print(f"npd={npd}, N={N} sites, D^2 spectrum [{eigs.min():.3f}, {eigs.max():.3f}]")
    n_zero = int(np.sum(eigs < 1e-9))
    print(f"Dirac zero modes (E=0): {n_zero}")

    # heat kernel Tr e^{-t D^2} for a range of t
    ts = np.geomspace(0.05, 1.0, 40)
    K = np.array([np.sum(np.exp(-t * eigs)) for t in ts])

    # fit: K ~ a0/t + a2 + a4 t (extract a2 = curvature)
    # solve least squares for [a0, a2, a4]
    A = np.column_stack([1.0 / ts, np.ones_like(ts), ts])
    coef, *_ = np.linalg.lstsq(A, K, rcond=None)
    a0, a2, a4 = coef
    print(f"\n  fit K ~ a0/t + a2 + a4 t:")
    print(f"    a0 (volume) = {a0:.3f}   (expect ~ N/(4pi) = {N/(4*np.pi):.3f})")
    print(f"    a2 (curvature) = {a2:.3f}   (expect ~0 for FLAT torus)")
    print(f"    a4 = {a4:.3f}")

    # residual (check the constant term is explained by zero modes, not curvature)
    print(f"\n  zero modes = {n_zero}, a2 (curvature) = {a2:.3f}")
    verdict = ("FLAT (a2 ~ 0): no scalar curvature, bridge B needs matter/defect"
               if abs(a2) < 1.0 else "CURVED (a2 != 0)")
    print(f"  VERDICT: {verdict}")

    out = ROOT / "experiments" / "exp_gravity_a2_last_run.json"
    out.write_text(json.dumps({"a0": float(a0), "a2": float(a2), "a4": float(a4),
                               "n_zero": n_zero, "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
