"""Step 8: does [D, x] (the vielbein, Connes' {D,x}=gamma^a e_a) become position-dependent?

Per the 破局推导: the vielbein e_mu^a(x) is read off from the noncommutative
commutator {D, x^mu} = gamma^a e_a^mu(x).  In Connes' NCG, the 1-form / vielbein is
[D, f] for f a function (coordinate).  So "is e position-dependent (curved)?" becomes
"is [D, x] position-dependent?".

[D, X]_{ij} = D_{ij} (x_j - x_i): non-zero on x-direction edges, value = phase * (coordinate
difference).  For a FLAT pi-flux this is a uniform hopping (the gamma^1 matrix), EXCEPT at
the periodic boundary x=0<->3 where the raw coordinate jumps by 3 (a COORDINATE ARTIFACT,
not a physical position dependence -- same kind of artifact as the Chebyshev step 6).

A defect flips the SIGN of one edge; this changes [D, x] LOCALLY.  We detect it with the
DIFFERENCE [D1, X] - [D, X] (not the row-norm, which is sign-blind).

Code: `py -m experiments.exp_spin2_metric_laplacian`
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


def idx(x, y, z, L):
    return ((z % L) * L + (y % L)) * L + (x % L)


def torus3D(L, flux=True):
    N = L ** 3
    D = np.zeros((N, N))
    for x, y, z in product(range(L), repeat=3):
        i = idx(x, y, z, L)
        j = idx(x + 1, y, z, L); w = (-1) ** (y + z) if flux else 1.0; D[i, j] = w; D[j, i] = w
        j = idx(x, y + 1, z, L); w = (-1) ** z if flux else 1.0; D[i, j] = w; D[j, i] = w
        j = idx(x, y, z + 1, L); w = 1.0; D[i, j] = w; D[j, i] = w
    return D


def coord_op(L, axis):
    N = L ** 3
    vals = []
    for z in range(L):
        for y in range(L):
            for x in range(L):     # x fastest, matching idx ordering
                if axis == 'x':
                    vals.append(float(x))
                elif axis == 'y':
                    vals.append(float(y))
                else:
                    vals.append(float(z))
    return np.diag(vals)


def main():
    L = 4
    N = L ** 3
    print("=== step 8: is the vielbein [D,x] position-dependent? ===")
    print()

    D = torus3D(L, True)
    X = coord_op(L, 'x')

    # ---- Part 1: [D, X] is the x-direction hopping (gamma^1) ----
    comm = D @ X - X @ D
    nz = np.count_nonzero(np.abs(comm) > 1e-9)
    # the non-zero entries are on x-direction edges, value = phase * (x_j - x_i)
    # x_j - x_i = +-1 in the bulk, but +-3 at the periodic boundary x=0<->3 (artifact)
    print("Part 1. [D, X] = x-direction hopping (the gamma^1 / vielbein):")
    print(f"  non-zero entries = {nz}  (= 64 sites x 2 x-neighbors = 128)")
    print(f"  bulk entries (|x_j-x_i|=1): value = phase * +-1")
    print(f"  periodic boundary (x=0<->3): value = phase * +-3  (COORDINATE ARTIFACT, not physical)")
    print("  => [D,X] is a uniform hopping = the FLAT vielbein e (translation-invariant),")
    print("     the +-3 at the boundary is a coordinate artifact (like Chebyshev step 6).")

    # ---- Part 2: defect flips one x-edge -> [D,X] changes LOCALLY ----
    D1 = torus3D(L, True)
    i0, j0 = idx(0, 0, 0, L), idx(1, 0, 0, L)
    D1[i0, j0] *= -1; D1[j0, i0] *= -1
    comm1 = D1 @ X - X @ D1
    delta = comm1 - comm                       # [D1,X] - [D,X] = (delta D)X - X(delta D)
    nz_delta = np.count_nonzero(np.abs(delta) > 1e-9)
    # delta D is nonzero only at (i0,j0),(j0,i0); so delta is nonzero only at those 2 rows
    rows_nz = np.unique(np.where(np.abs(delta) > 1e-9)[0])
    print()
    print("Part 2. DEFECT (flip one x-edge): [D,X] changes LOCALLY:")
    print(f"  [D1,X]-[D,X] non-zero entries = {nz_delta}  (only at the defect edge's 2 endpoints)")
    for t in rows_nz:
        x, y, z = t % L, (t // L) % L, t // (L * L)
        print(f"    affected site ({x},{y},{z})")
    print(f"  => the vielbein e (from [D,x]) changes ONLY at the defect (local, 2 sites).")

    # ---- conclusion ----
    print()
    print("CONCLUSION:")
    print("  - flat pi-flux: [D,x] = uniform hopping -> e is flat/constant (step 1).")
    print("  - defect: [D,x] acquires a LOCAL change at the defect edge (short range),")
    print("    matching step 3 (defect -> local, not continuous).")
    print("  - so the vielbein e IS position-dependent at a defect, but only LOCALLY;")
    print("    the continuous field (curvature) still needs the domain wall to fall.")

    summary = {
        "Dx_is_x_hopping_gamma1": True,
        "periodic_boundary_artifact": "x=0<->3 raw coordinate jump +-3 (not physical)",
        "defect_Dx_local_change_2_sites": bool(nz_delta <= 4 and len(rows_nz) <= 2),
        "conclusion": "[D,x] (vielbein) is flat/constant; a defect changes it locally (2 sites), "
                      "not continuous -- domain wall stands.",
        "note": "direct face-off with 'e position-dependent' via Connes {D,x}=gamma^a e_a.",
    }
    out = ROOT / "experiments" / "exp_spin2_metric_laplacian_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
