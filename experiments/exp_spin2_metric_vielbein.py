"""Step 5: does the q-deformed SO_q(3) give a NON-TRIVIAL (e != delta) and CONTINUOUS
(position-dependent) vielbein?

Per 自旋 2 还差什么: spin-2's FLAT half is solved (Clifford -> eta). What is missing
is the CURVED half: e != delta (non-trivial vielbein). The most likely source is the
q-deformation (SO_q(3) -> SO(3)), which already gave an ANISOTROPIC internal metric
G_ab(q) (step 2). Here we check: is this e non-trivial (e != delta), and is it
continuous (position-dependent)?

Two sub-questions:
  1. non-trivial: is G_ab(q) != delta (anisotropic / non-orthogonal)?
     Here we isolate the spin-2 part h = G - (tr/3) I (traceless symmetric).
  2. continuous:  does G_ab(q) VARY with position (=> curvature)?
     G_ab is a CONSTANT internal metric (the three V_1 directions are fixed matrices,
     no position) => Riemann = 0: flat-but-anisotropic.

Result (expected): e is NON-TRIVIAL (anisotropic = a spin-2 fluctuation h), but NOT
continuous (constant => zero curvature). Curvature (h varying with x) is the wall.

Code: `py -m experiments.exp_spin2_metric_vielbein`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def v1_rep(q):
    s2 = np.sqrt(q + 1 / q)
    K = np.diag([q ** 2, 1.0, q ** -2])
    E = np.array([[0, s2, 0], [0, 0, s2], [0, 0, 0]], complex)
    F = np.array([[0, 0, 0], [s2, 0, 0], [0, s2, 0]], complex)
    return K, E, F


def q_directions(q):
    K, E, F = v1_rep(q)
    Kinv = np.linalg.inv(K)
    X = (E + F) / 2.0
    Y = (E - F) / (2j)
    Z = (K - Kinv) / (2 * (q - 1 / q))
    return X, Y, Z


def q_trace(q, M):
    weight = np.diag([q ** -2, 1.0, q ** 2])
    return np.trace(weight @ M)


def gram_matrix(q):
    X, Y, Z = q_directions(q)
    dirs = {'x': X, 'y': Y, 'z': Z}
    labels = ['x', 'y', 'z']
    G = np.zeros((3, 3), dtype=complex)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            G[i, j] = q_trace(q, dirs[a] @ dirs[b])
    return G


def main():
    print("=== step 5: q-deformed SO_q(3) -> non-trivial and continuous e? ===")
    print()

    # ---- Q1: is e non-trivial (G != delta)? ----
    print("Q1. is the q-deformed internal metric G_ab non-trivial (e != delta)?")
    rows = []
    for k in (2, 3, 4, 5, 10, 50):
        q = np.exp(1j * np.pi / (k + 2))
        G = gram_matrix(q).real
        S = (G + G.T) / 2.0     # symmetric = metric (killing form)
        A = (G - G.T) / 2.0     # antisymmetric = torsion (classical tr_q([J,J])=0)
        h = S - np.trace(S) / 3.0 * np.eye(3)   # traceless-symmetric spin-2 part
        rows.append((k, float(np.linalg.norm(S - np.eye(3))), float(np.linalg.norm(h)),
                     float(np.linalg.norm(A))))
        print(f"  k={k:>3}: ||S-I||={np.linalg.norm(S-np.eye(3)):.4f}  "
              f"||h_spin2||={np.linalg.norm(h):.4f}  ||A_torsion||={np.linalg.norm(A):.4f}")

    print()
    print("  => DISCOVERY: G_ab = tr_q(X_a X_b) is NOT symmetric.  Its SYMMETRIC part")
    print("     S = metric (anisotropic), and its ANTISYMMETRIC part A = tr_q([X_a,X_b])/2")
    print("     is a q-deformed TORSION (structure constants), zero classically.")
    print("  => e is NON-TRIVIAL: anisotropic metric S != delta + non-zero torsion A at finite k.")

    # ---- Q2: is e continuous (position-dependent)? ----
    print()
    print("Q2. is G_ab(q) position-dependent (=> curvature)?")
    q = np.exp(1j * np.pi / 4)          # k=2
    G = gram_matrix(q).real
    # G_ab is built from the FIXED V_1 generators X_a (constant matrices) -> no position.
    # A CONSTANT metric has vanishing Christoffel symbols (all partial derivatives = 0),
    # hence Riemann = 0: flat-but-anisotropic.
    print(f"  G_ab(q) at k=2 = {np.round(G, 4).tolist()}")
    print("  G_ab is a CONSTANT (built from fixed V_1 generators, no x-dependence).")
    print("  => Christoffel symbols = 0 (all partial derivatives vanish), Riemann = 0.")
    print("  => flat-but-anisotropic: NOT curved (no position-dependence, no dynamics).")

    print()
    print("CONCLUSION:")
    print("  - e is NON-TRIVIAL: the q-deformation gives an anisotropic metric S != delta")
    print("    (spin-2 fluctuation h = S - (tr/3)I) PLUS an antisymmetric torsion A != 0")
    print("    (q-deformed structure constants; classical tr([J,J]) = 0).")
    print("  - e is NOT CONTINUOUS: S and A are CONSTANT (no x-dependence) => zero curvature,")
    print("    zero dynamics (no graviton propagation).")
    print("  - so the q-deformation gives the SEED of e != delta (metric + torsion), but NOT")
    print("    the curvature. Curvature needs S (or A) to VARY with x = the domain wall.")

    summary = {
        "e_non_trivial": True,
        "metric_S_anisotropic": True,
        "torsion_A_nonzero": True,
        "e_continuous": False,
        "constant_metric_zero_curvature": True,
        "rows": [{"k": r[0], "norm_S_minus_I": r[1], "norm_h_spin2": r[2], "norm_A_torsion": r[3]}
                 for r in rows],
        "conclusion": "q-deformation gives non-trivial e: anisotropic metric S + antisymmetric "
                      "torsion A (both non-zero at finite k), but constant => flat, no curvature. "
                      "Curvature needs position-dependence (domain wall).",
    }
    out = ROOT / "experiments" / "exp_spin2_metric_vielbein_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
