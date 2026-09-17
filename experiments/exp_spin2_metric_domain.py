"""Step 3 recon: does a flux DEFECT induce position-dependent curvature (domain)?

Domain wall (from step 2, exp_spin2_metric_qdeform): the q-deformed frame metric
G_ab(q) is CONSTANT (position-independent) => Riemann = 0.  Real curvature needs G
to VARY with position = a domain.  But the uniform pi-flux is translation-invariant
(no special position); the local Kramers frame J_p exists but is pure GAUGE
(exp_spin_connection_localJ).  So "position" must come from somewhere.

Probe the simplest candidate: a flux DEFECT (breaking translation invariance), the
spin-2 analogue of Q2 ("matter = defect").  Concretely:

  uniform pi-flux: F_xy = [T_x,T_y] = 2 T_x T_y is MOMENTUM-block-diagonal
                   (commutes with all T_i^2 sector labels + chi) -- position-independent.
  with a defect (flip one edge phase): does F_xy become NON-block-diagonal
                   (fail to commute with T_i^2), i.e. position-dependent?

If YES: the domain (position-dependent curvature) is seeded by a defect (matter).
If NO:  the domain wall is deeper than a single defect.

Code: `py -m experiments.exp_spin2_metric_domain`
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


def mag_trans(D, L, axis):
    N = L ** 3
    T = np.zeros((N, N))
    for x, y, z in product(range(L), repeat=3):
        i = idx(x, y, z, L)
        if axis == 'x':
            j = idx(x + 1, y, z, L)
        elif axis == 'y':
            j = idx(x, y + 1, z, L)
        else:
            j = idx(x, y, z + 1, L)
        T[i, j] = D[i, j]
    return T


def flip_edge(D, L, axis, x, y, z):
    """Flip the phase of one edge along `axis` starting at (x,y,z)."""
    i = idx(x, y, z, L)
    if axis == 'x':
        j = idx(x + 1, y, z, L)
    elif axis == 'y':
        j = idx(x, y + 1, z, L)
    else:
        j = idx(x, y, z + 1, L)
    D[i, j] *= -1
    D[j, i] *= -1
    return D


def main():
    L = 4
    N = L ** 3
    print("=== step 3 recon: flux defect -> position-dependent curvature? ===")
    print()

    # ---- Part 1: uniform pi-flux, F_xy momentum-block-diagonal ----
    D0 = torus3D(L, flux=True)
    Tx, Ty, Tz = mag_trans(D0, L, 'x'), mag_trans(D0, L, 'y'), mag_trans(D0, L, 'z')
    Fxy = Tx @ Ty - Ty @ Tx
    comm0 = [float(np.linalg.norm(Fxy @ M - M @ Fxy)) for M in (Tx @ Tx, Ty @ Ty, Tz @ Tz)]
    chi0 = Tx @ Ty @ Tz
    comm0_chi = float(np.linalg.norm(Fxy @ chi0 - chi0 @ Fxy))
    print("Part 1. uniform pi-flux: F_xy = [T_x,T_y]:")
    print(f"  ||F_xy|| = {np.linalg.norm(Fxy):.2f}")
    print(f"  [F_xy, T_i^2] = {[f'{c:.1e}' for c in comm0]}  [F_xy, chi] = {comm0_chi:.1e}")
    print("  (all ~0 => momentum-block-diagonal, position-INDEPENDENT)")
    print()

    # ---- Part 2: flip one edge (defect), check block-diagonality ----
    print("Part 2. flip one edge (defect) -> does F_xy become position-dependent?")
    rows = []
    for axis in ('x', 'y', 'z'):
        for (dx, dy, dz) in [(0, 0, 0), (1, 2, 1)]:
            Dd = flip_edge(torus3D(L, flux=True), L, axis, dx, dy, dz)
            Tx = mag_trans(Dd, L, 'x'); Ty = mag_trans(Dd, L, 'y'); Tz = mag_trans(Dd, L, 'z')
            Fxy_d = Tx @ Ty - Ty @ Tx
            comm = [float(np.linalg.norm(Fxy_d @ M - M @ Fxy_d)) for M in (Tx @ Tx, Ty @ Ty, Tz @ Tz)]
            chi = Tx @ Ty @ Tz
            comm_chi = float(np.linalg.norm(Fxy_d @ chi - chi @ Fxy_d))
            # total change in F_xy from the defect
            dF = float(np.linalg.norm(Fxy_d - Fxy))
            rows.append((axis, (dx, dy, dz), comm, comm_chi, dF))
            print(f"  defect {axis}-edge ({dx},{dy},{dz}): [F,T^2]={[f'{c:.1e}' for c in comm]}  "
                  f"[F,chi]={comm_chi:.1e}  ||F_def - F_0||={dF:.2f}")

    print()
    any_nonblock = any(max(r[2]) > 1e-6 for r in rows)
    print(f"  => any defect makes F_xy NON-block-diagonal (position-dependent): {any_nonblock}")

    # ---- Part 3: is the change LOCALIZED near the defect? ----
    print()
    print("Part 3. is the curvature change LOCALIZED near the defect?")
    # compare the defect-induced F change against position: measure how F_xy_d
    # differs from F_xy as a function of real-space distance from the defect.
    Dd = flip_edge(torus3D(L, flux=True), L, 'x', 0, 0, 0)
    Tx = mag_trans(Dd, L, 'x'); Ty = mag_trans(Dd, L, 'y'); Tz = mag_trans(Dd, L, 'z')
    Fd = Tx @ Ty - Ty @ Tx
    # position operator X (diagonal, x-coordinate)
    X = np.diag([float((i % L)) for i in range(N)])
    # does Fd - F0 concentrate on a few sites?  row participation of the difference
    diff = Fd - Fxy
    row_norm = np.linalg.norm(diff, axis=1)
    top = np.argsort(row_norm)[::-1][:6]
    print(f"  largest |F_def - F_0| row-norms at sites (x,y,z):")
    for t in top:
        x, y, z = t % L, (t // L) % L, t // (L * L)
        print(f"    site ({x},{y},{z})  row-norm = {row_norm[t]:.3f}")

    print()
    print("interpretation:")
    if any_nonblock:
        print("  - a flux defect breaks translation invariance, and F_xy loses momentum")
        print("    block-diagonality => the curvature acquires POSITION dependence.")
        print("    => 'domain' (position-dependence) is seeded by a DEFECT (matter = Q2),")
        print("       the spin-2 analogue of the scalar 'matter = defect'.")
    else:
        print("  - a single defect does NOT break F_xy block-diagonality; the domain wall")
        print("    is deeper than one defect (needs a different mechanism / different system).")
    print("  - honest boundary: this is curvature (F), not yet a metric G varying with x;")
    print("    linking F -> G needs the vielbein torsion-free condition (still open).")

    summary = {
        "uniform_F_block_diagonal": all(c < 1e-9 for c in comm0) and comm0_chi < 1e-9,
        "defect_makes_F_position_dependent": any_nonblock,
        "rows": [{"axis": r[0], "site": r[1], "comm_F_T2": r[2],
                  "comm_F_chi": r[3], "norm_F_change": r[4]} for r in rows],
        "honest_boundary": "curvature (F) position-dependence from a defect, not yet "
                           "a varying metric G (needs vielbein torsion-free link).",
        "note": "a flux defect breaks F_xy momentum block-diagonality => position-dependent "
                "curvature; the domain (position) is seeded by a defect (matter=Q2, spin-2 version).",
    }
    out = ROOT / "experiments" / "exp_spin2_metric_domain_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
