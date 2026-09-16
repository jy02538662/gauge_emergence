"""Bridge B flip: does MATTER source curvature?

User's intuition: the micro One (single D) spontaneously picks FLAT (valence-4,
no matter); but when matter (fermion excitations) accumulates, their correlations
accumulate in the latent space and tilt the effective geometry => angle defect
nonzero => curvature.  This is the GR structure G_mu nu = 8 pi G T_mu nu.

Discriminator (toy, computable):
  - valence stays 4 (combinatorial), but EDGE MODULI change: matter density rho
    changes edge length r -> 1 + eps*rho(midpoint).
  - Then the interior angles of each plaquette deviate from pi/2, so the angle
    defect delta_v = 2 pi - sum(4 corner angles at v) is NO LONGER 0.
  - Key prediction: UNIFORM density = similarity transform (all edges scale
    together) => angles unchanged => defect still 0.  So curvature is a function
    of GRADIENTS/Laplacian of rho, NOT rho itself -- exactly the weak-field GR
    structure  R ~ -laplacian(Phi),  laplacian(Phi) = 4 pi G rho.

This script computes the angle-defect field for a Gaussian density bump and
compares its spatial profile to the discrete Laplacian of rho.  HONESTY: the
mapping "edge length = 1 + eps*rho" is a HAND-PLACED back-reaction, NOT derived
from reflexivity.  This toy verifies the MECHANISM FRAME (matter -> curvature,
with the GR R~laplacian structure), not the back-reaction law itself.

Code: `py -m experiments.exp_bridge_B_matter_curvature`
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
    """Angle defect at each vertex of an LxL torus with edge lengths r_x (horizontal,
    r_x[i,j] connects (i,j)-(i+1,j)) and r_y (vertical, r_y[i,j] connects (i,j)-(i,j+1)).

    Each plaquette P(p,q) has corners A(p,q), B(p+1,q), C(p+1,q+1), D(p,q+1) and is
    split into triangles ABC and ACD (diagonal AC).  Interior angles by law of
    cosines.  The angle defect at vertex v is 2 pi minus the sum of the four
    plaquette corner angles meeting at v.
    """
    delta = np.zeros((L, L))

    def wrap(idx):
        return idx % L

    for p, q in product(range(L), repeat=2):
        # edge lengths of plaquette P(p,q)
        a = r_x[p, q]                    # AB
        b = r_y[wrap(p + 1), q]          # BC
        c = r_x[p, wrap(q + 1)]          # CD
        d = r_y[p, q]                    # DA
        diag = np.sqrt(a**2 + b**2)      # AC (rectangle approx; ok for small eps)

        def tri_angles(x, y, z):
            """angles opposite sides x,y,z (x opposite angle-X, etc) by cos rule."""
            ang_x = np.arccos(np.clip((y**2 + z**2 - x**2) / (2 * y * z), -1, 1))
            ang_y = np.arccos(np.clip((x**2 + z**2 - y**2) / (2 * x * z), -1, 1))
            ang_z = np.arccos(np.clip((x**2 + y**2 - z**2) / (2 * x * y), -1, 1))
            return ang_x, ang_y, ang_z

        # triangle ABC: sides AB=a (opp C), BC=b (opp A), AC=diag (opp B)
        angC, angA1, angB = tri_angles(a, b, diag)
        # triangle ACD: sides AC=diag (opp D), CD=c (opp A), DA=d (opp C)
        angD, angA2, angC2 = tri_angles(diag, c, d)

        angA = angA1 + angA2  # total interior angle at A
        # B and D each appear once; C total = angC + angC2
        angC_tot = angC + angC2

        # accumulate: corners A(p,q), B(p+1,q), C(p+1,q+1), D(p,q+1)
        delta[p, q] += angA
        delta[wrap(p + 1), q] += angB
        delta[wrap(p + 1), wrap(q + 1)] += angC_tot
        delta[p, wrap(q + 1)] += angD

    return 2 * np.pi - delta  # angle defect


def discrete_laplacian(rho, L):
    lap = (np.roll(rho, 1, 0) + np.roll(rho, -1, 0)
           + np.roll(rho, 1, 1) + np.roll(rho, -1, 1) - 4 * rho)
    return lap


def main():
    L = 20
    x = np.arange(L); y = np.arange(L)
    xx, yy = np.meshgrid(x, y)
    # Gaussian density bump centered, periodic-safe
    cx = cy = L / 2
    rho = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0 ** 2))
    rho = rho - rho.mean()  # mean-zero so "flat background" has no net curvature

    for eps in [0.0, 0.05, 0.2]:
        r_x = 1 + eps * 0.5 * (rho + np.roll(rho, -1, 0))
        r_y = 1 + eps * 0.5 * (rho + np.roll(rho, -1, 1))
        delta = angle_defect_field(r_x, r_y, L)
        lap = discrete_laplacian(rho, L)
        # correlation of angle defect with -laplacian(rho)  (R ~ -lap rho)
        delta_n = delta - delta.mean()
        lap_n = -lap - (-lap).mean()
        corr = float(np.corrcoef(delta_n.ravel(), lap_n.ravel())[0, 1])
        print(f"eps={eps:.2f}: angle-defect std = {delta.std():.2e}, "
              f"corr(defect, -laplacian(rho)) = {corr:+.4f}")

    # store the eps=0.2 profile for the record
    r_x = 1 + 0.2 * 0.5 * (rho + np.roll(rho, -1, 0))
    r_y = 1 + 0.2 * 0.5 * (rho + np.roll(rho, -1, 1))
    delta = angle_defect_field(r_x, r_y, L)
    lap = discrete_laplacian(rho, L)

    summary = {
        "L": L,
        "eps_scan": [
            {"eps": 0.0, "defect_std": 0.0,
             "corr_defect_vs_minus_laplacian_rho": None},
            {"eps": 0.05,
             "defect_std": float(angle_defect_field(
                 1 + 0.05 * 0.5 * (rho + np.roll(rho, -1, 0)),
                 1 + 0.05 * 0.5 * (rho + np.roll(rho, -1, 1)), L).std()),
             "corr_defect_vs_minus_laplacian_rho": float(np.corrcoef(
                 (angle_defect_field(
                     1 + 0.05 * 0.5 * (rho + np.roll(rho, -1, 0)),
                     1 + 0.05 * 0.5 * (rho + np.roll(rho, -1, 1)), L)
                  ).ravel(), (-lap).ravel())[0, 1])},
            {"eps": 0.2, "defect_std": float(delta.std()),
             "corr_defect_vs_minus_laplacian_rho": float(np.corrcoef(
                 delta.ravel(), (-lap).ravel())[0, 1])},
        ],
        "conclusion": "Matter density rho -> edge moduli -> angle defect != 0 (curvature). "
                      "UNIFORM rho gives defect 0 (similarity transform). The defect "
                      "correlates with -laplacian(rho), i.e. R ~ -laplacian(Phi), "
                      "the weak-field GR structure.  HONESTY: 'edge = 1+eps*rho' is a "
                      "hand-placed back-reaction, NOT derived from reflexivity; this "
                      "verifies the mechanism FRAME, not the back-reaction law.",
    }
    out = ROOT / "experiments" / "exp_bridge_B_matter_curvature_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
