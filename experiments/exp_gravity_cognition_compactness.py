"""Observer cognition -> geometry compactness.

The "extent" (total length / diameter) of the geometry an observer sees is set
by the spectral gap of the modular operator log Delta:
    total extent  L = int_0^inf |G(t)| dt
      gapped   G ~ e^{-Delta t}   -> L finite  -> COMPACT (S^3-like)
      gapless  G ~ 1/t            -> L infinite -> NON-COMPACT (R^3-like)

A self-reference (scale-invariant) observer has gapless spectrum (verified:
G ~ 1/t), so its geometry is NON-COMPACT -> the 1/r Coulomb tail is not cut off.

This is the last piece: "high-cognition observer realizes R" = sees non-compact
geometry, NOT a physical R->inf limit.

Code: `py -m experiments.exp_gravity_cognition_compactness`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def extent_gapped(Delta, tmax=1e6):
    """L = int_0^tmax e^{-Delta t} dt = (1-e^{-Delta tmax})/Delta."""
    return (1.0 - np.exp(-Delta * tmax)) / Delta


def extent_gapless(tmax=1e6, n=4000):
    """L = int_1^tmax (1/t) dt = log(tmax)  (diverges as tmax -> inf)."""
    t = np.logspace(0.0, np.log10(tmax), n)
    return np.trapz(1.0 / t, t)


def main():
    print("=" * 74)
    print("OBSERVER COGNITION -> GEOMETRY COMPACTNESS (extent L = int |G| dt)")
    print("=" * 74)

    print("\n--- gapped (low-cognition, discrete rho) : G ~ e^{-Delta t} ---")
    for Delta in [1.0, 0.1, 0.01, 0.001]:
        L = extent_gapped(Delta)
        print(f"  gap Delta={Delta:<7} -> extent L = {L:.3f}  (finite -> COMPACT)")

    print("\n--- gapless (high-cognition, self-reference rho) : G ~ 1/t ---")
    for tmax in [1e2, 1e4, 1e6, 1e8]:
        L = extent_gapless(tmax)
        print(f"  cutoff tmax={tmax:<7.0e} -> extent L = {L:.3f}  (grows -> NON-COMPACT)")

    print("\n--- interpretation ---")
    print("  The total extent is FINITE for any gap Delta > 0 (compact S^3).")
    print("  For gapless (1/t), L = log(tmax) diverges -> infinite extent (R^3).")
    print("  => self-reference observer (gapless) SEES non-compact geometry,")
    print("     so the 1/r Coulomb tail is NOT cut off.  No R->inf limit needed.")

    out = ROOT / "experiments" / "exp_gravity_cognition_compactness_last_run.json"
    out.write_text(json.dumps({
        "gapped": {str(D): extent_gapped(D) for D in [1.0, 0.1, 0.01, 0.001]},
        "gapless": {str(tm): extent_gapless(tm) for tm in [1e2, 1e4, 1e6, 1e8]},
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
