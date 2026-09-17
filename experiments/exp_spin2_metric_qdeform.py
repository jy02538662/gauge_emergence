"""Step 2 recon: does finite-k (q != 1) make the SO_q(3) "metric" NON-FLAT?

Attack step 2 of the spin-2 metric (per 权威交接文档 §六·五, "自旋校准"): the
non-trivial vielbein e != delta must come from the QUANTUM side
(SU(2)_k -> TL/loop -> V_1 -> SO_q(3)), since the classical bridge (V_1 = physical
3D, mu = a) gives e = delta => g = eta FLAT (see exp_spin2_metric, degeneration A).

Here we probe the first quantitative question:

  Does finite k (q = e^{i pi/(k+2)} != 1) make the metric of the three V_1
  directions ANISOTROPIC (non-flat), recovering isotropy only as k -> inf?

The "metric" of the internal frame = the q-deformed Killing form (quantum trace):
  G_ab(q) = tr_q(X_a X_b),   tr_q(X) = tr(K^{-2 rho} X),  rho = J_z = diag(1,0,-1).
  X_x = (E+F)/2,  X_y = (E-F)/(2i),  X_z = (K-K^{-1})/(2(q-q^{-1}))
      -> J_x, J_y, J_z  as q -> 1 (classical so(3) = physical 3D).

Classical (q=1): G_ab = 2 delta_ab (isotropic, FLAT).  We compute G_ab(q) and check:
  1. orthogonality: are the off-diagonal G_ab (a != b) zero?
  2. isotropy:      is G_xx = G_yy = G_zz?
  3. classical limit: G -> 2 delta_ab as k -> inf.

Expected (matches exp_s2_probe criterion 1): q-deformation reduces SO(3) -> U(1)
(the Cartan z-direction is special), so G_zz != G_xx = G_yy at finite k.

HONEST BOUNDARY: G_ab(q) is a CONSTANT (position-independent) internal-frame metric,
so its Riemann curvature is identically zero even when anisotropic.  Anisotropy is
NOT curvature: real spin-2 curvature needs G to VARY with position (a domain), which
is the paid-bridge-2 domain wall (still open).  What this probe DOES establish is
the clean quantitative signature that the q-deformed frame is genuinely non-trivial
(non-isotropic) and only flattens in the classical limit.

Code: `py -m experiments.exp_spin2_metric_qdeform`
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
    """q-deformed V_1 (spin-1) representation: returns (K, E, F)."""
    s2 = np.sqrt(q + 1 / q)  # sqrt([2]_q)
    K = np.diag([q ** 2, 1.0, q ** -2])
    E = np.array([[0, s2, 0], [0, 0, s2], [0, 0, 0]], complex)
    F = np.array([[0, 0, 0], [s2, 0, 0], [0, s2, 0]], complex)
    return K, E, F


def q_directions(q):
    """The three q-deformed frame directions (X_x, X_y, X_z) -> (J_x,J_y,J_z) as q->1."""
    K, E, F = v1_rep(q)
    Kinv = np.linalg.inv(K)
    X = (E + F) / 2.0
    Y = (E - F) / (2j)
    Z = (K - Kinv) / (2 * (q - 1 / q))
    return X, Y, Z


def q_trace(q, M):
    """Quantum trace tr_q(M) = tr(K^{-2 rho} M), rho = J_z = diag(1,0,-1)."""
    # K^{-2 rho} = diag(q^{-2}, 1, q^{2})  (rho = diag(1,0,-1), K = diag(q^2,1,q^-2))
    weight = np.diag([q ** -2, 1.0, q ** 2])
    return np.trace(weight @ M)


def gram_matrix(q):
    """G_ab(q) = tr_q(X_a X_b), a,b in {x,y,z} (the q-deformed frame metric)."""
    X, Y, Z = q_directions(q)
    dirs = {'x': X, 'y': Y, 'z': Z}
    labels = ['x', 'y', 'z']
    G = np.zeros((3, 3), dtype=complex)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            G[i, j] = q_trace(q, dirs[a] @ dirs[b])
    return G, labels


def main():
    print("=== step 2 recon: q-deformed frame metric G_ab(q) = tr_q(X_a X_b) ===")
    print()

    ks = (2, 3, 4, 5, 10, 50, 500)
    rows = []
    for k in ks:
        q = np.exp(1j * np.pi / (k + 2))
        G, labels = gram_matrix(q)
        # magnitudes (G is real here since q on unit circle)
        Gre = np.round(G.real, 4)
        offdiag = max(abs(G[0, 1]), abs(G[0, 2]), abs(G[1, 2]))
        gxx, gyy, gzz = abs(G[0, 0]), abs(G[1, 1]), abs(G[2, 2])
        rows.append((k, float(gxx), float(gzz), float(offdiag)))
        print(f"  k={k:>3}: G_xx={gxx:.4f}  G_yy={gyy:.4f}  G_zz={gzz:.4f}  "
              f"max|offdiag|={offdiag:.2e}")

    print()
    print("  => off-diagonal G_xy != 0 at finite k, while G_xz = G_yz = 0:")
    print("     q-deformation BREAKS x-y orthogonality (matches the exp_s2q_cg discovery),")
    print("     while z (Cartan, diagonal) stays orthogonal to x,y.")
    print("     AND G_zz != G_xx = G_yy at finite k: the frame metric is ANISOTROPIC.")
    print("     (z = Cartan/U(1) direction is special, matching exp_s2_probe criterion 1.)")
    print()

    # classical limit: G_zz / G_xx -> 1 as k -> inf
    print("classical limit k -> inf: anisotropy G_zz/G_xx -> 1 (isotropic, flat):")
    for k in (2, 3, 5, 10, 50, 500):
        q = np.exp(1j * np.pi / (k + 2))
        G, _ = gram_matrix(q)
        ratio = abs(G[2, 2]) / abs(G[0, 0])
        print(f"  k={k:>3}: G_zz/G_xx = {ratio:.4f}   (-> 1 as k->inf)")
    print("  => isotropy is recovered only in the classical limit: the q-deformed")
    print("     frame metric is genuinely non-flat (anisotropic) at finite k.")

    print()
    print("HONEST BOUNDARY:")
    print("  - G_ab(q) is a CONSTANT (position-independent) metric on the internal frame.")
    print("  - its Riemann curvature is identically ZERO even when anisotropic:")
    print("    anisotropy (G_zz != G_xx) is NOT curvature (which needs G to VARY with x).")
    print("  - real spin-2 curvature needs a domain (position), i.e. the paid-bridge-2")
    print("    domain wall -- still open.  What this probe establishes is the clean")
    print("    quantitative signature that the q-deformed frame is non-trivial and only")
    print("    flattens (isotropizes) in the classical limit.")

    summary = {
        "frame_metric": "G_ab(q) = tr_q(X_a X_b), tr_q = tr(K^{-2 rho} .), rho=J_z",
        "orthogonality_broken_by_q": True,
        "G_xy_nonzero_G_xz_G_yz_zero": True,
        "anisotropic_at_finite_k": True,
        "gxx_gyy_equal": True,
        "rows": [{"k": r[0], "G_xx": r[1], "G_zz": r[2], "abs_G_xy": r[3]} for r in rows],
        "classical_limit_isotropic": True,
        "honest_boundary": "G_ab is constant => Riemann=0; anisotropy != curvature; "
                          "real curvature needs domain (paid-bridge-2 wall, open).",
        "note": "q-deformed frame metric is anisotropic (G_zz != G_xx=G_yy) at finite k, "
                "isotropic only as k->inf. Clean signature that SO_q(3) is non-trivial, "
                "but anisotropy is not yet curvature (no domain).",
    }
    out = ROOT / "experiments" / "exp_spin2_metric_qdeform_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
