"""Step 10: the 'no characteristic scale -> long range' criterion, made explicit.

Per 双自指 (self-referential D x self-referential rho): the ONLY thing that gives
long range is "no characteristic scale" (scale invariance).  But there are TWO sources:

  - rho's scale invariance: EMERGENT (uniqueness A: no external observer -> no preference
    -> rho = C/lambda), already gives 1/t -> 1/r.
  - D's self-similarity:   INPUT (a hand-placed power-law / fractal hopping).

Here we make the criterion explicit on the vielbein [D, X]:
  - power-law hopping  D_ij = 1/|i-j|^alpha  (no characteristic scale) -> [D,X] ~ 1/r^(alpha-1)
    => LONG range (power law)
  - nearest-neighbor hopping (characteristic scale = lattice spacing) -> [D,X] = 0 for r>1
    => SHORT range (local)

This confirms "long range = no characteristic scale", but HONESTLY: the power-law D is an
INPUT, not an emergence.  The emergent scale-invariance (uniqueness A) lives in rho, not in
D; the self-referential D (candidates A/B/C) loops back to short range.

Code: `py -m experiments.exp_spin2_metric_selfref`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def comm_DX(D, X):
    return D @ X - X @ D


def main():
    print("=== step 10: 'no characteristic scale -> long range' criterion (vielbein) ===")
    print()

    N = 128
    X = np.diag(np.arange(N, dtype=float))

    # ---- power-law hopping (no characteristic scale) ----
    alpha = 2.0
    D_pl = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                D_pl[i, j] = 1.0 / abs(i - j) ** alpha
    C_pl = comm_DX(D_pl, X)                       # [D,X]_ij = (j-i)/|i-j|^alpha ~ 1/r^(alpha-1)
    # matrix element magnitude vs distance r
    rs = np.arange(1, N)
    mag_pl = np.array([abs(C_pl[0, r]) for r in rs])

    # ---- nearest-neighbor hopping (characteristic scale = 1) ----
    D_nn = np.zeros((N, N))
    for i in range(N - 1):
        D_nn[i, i + 1] = 1.0
        D_nn[i + 1, i] = 1.0
    C_nn = comm_DX(D_nn, X)
    mag_nn = np.array([abs(C_nn[0, r]) for r in rs])

    print("vielbein [D,X] matrix element |[D,X]_{0,r}| vs distance r:")
    print(f"  {'r':>4} {'power-law(1/r)':>16} {'nearest-neighbor':>18}")
    for r in (1, 2, 4, 8, 16, 32, 64, 127):
        print(f"  {r:>4} {mag_pl[r-1]:>16.5f} {mag_nn[r-1]:>18.5f}")

    # fit power law of power-law hopping
    keep = rs[3:60]
    p_pl = float(np.polyfit(np.log(keep), np.log(mag_pl[keep - 1] + 1e-30), 1)[0])
    print()
    print(f"  power-law hopping [D,X]: slope of log|[D,X]| vs log r = {p_pl:.3f}  (power law, long)")
    print(f"  nearest-neighbor [D,X]:  nonzero only at r=1  (local, short)")

    print()
    print("CONCLUSION:")
    print("  - no characteristic scale (power-law hopping) -> [D,X] is LONG range (power law),")
    print("    BUT TRANSLATION-INVARIANT (depends only on |i-j|, not on absolute position i).")
    print("  - a characteristic scale (nearest neighbor) -> [D,X] is LOCAL.")
    print("  - KEY DISTINCTION: 'long range' and 'curvature' are TWO ORTHOGONAL axes:")
    print("      * long range   = no characteristic scale (power law / scale invariance)")
    print("      * curvature    = position-DEPENDENCE (breaks translation invariance, needs a defect)")
    print("    The power-law D gives LONG range but FLAT (translation-invariant, e = constant).")
    print("    A defect gives curvature but SHORT range.  No construction gives BOTH yet.")
    print("  - so 'long range = no characteristic scale' is CONFIRMED, but it is NOT curvature,")
    print("    and the power-law D is an INPUT, not an emergence.")

    summary = {
        "power_law_D_long_range": bool(p_pl < -0.5),
        "nn_D_short_range": bool(mag_nn[1] < 1e-12),
        "slope_power_law": p_pl,
        "long_range_is_flat_not_curved": True,
        "key_distinction": "long range (scale invariance, translation-invariant) vs curvature "
                           "(position-dependence, needs defect) are two ORTHOGONAL axes.",
        "honest": "long range = no characteristic scale (confirmed), but power-law D is an INPUT "
                  "and gives FLAT (translation-invariant), not curved.",
        "note": "self-referential D (A/B/C) loops back short range; input power-law D gives long "
                "range but flat; no construction gives long+curved yet.",
    }
    out = ROOT / "experiments" / "exp_spin2_metric_selfref_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
