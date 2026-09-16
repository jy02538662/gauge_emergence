"""Read-C: does "no preferred direction" act on CURVATURE (not moduli)?

Puzzle 1 (grad^2 mismatch):  reaction r ~ rho gives delta ~ grad^2 rho, but GR
needs delta ~ rho.  Read C resolves it: "no preferred direction" (= S_N invariance)
should act on the CURVATURE delta_v (S_N-invariant scalar), NOT on the moduli
r_ij (S_N-variant).  Then delta ~ rho directly = GR field equation.

Concretely, the back-reaction should be  r ~ Phi = grad^-2 rho  (potential,
Poisson inverse), NOT  r ~ rho  (density, contact).  Then:
    delta ~ grad^2 r ~ grad^2 Phi ~ rho   (GR field equation trace)

Test: on a periodic torus, build a density bump rho, solve grad^2 Phi = rho,
and compute the angle defect delta for BOTH back-reactions:
  (A/B) r = 1 + alpha * (endpoint DENSITY sum)   -> expect delta ~ grad^2 rho
  (C)   r = 1 + alpha * (endpoint POTENTIAL sum) -> expect delta ~ rho
Report correlation of delta with rho and with -grad^2 rho in each case.

Code: `py -m experiments.exp_bridge_B_weld_readC`
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
    """Regge angle defect.  r_x[i,j] horizontal edge (i,j)-(i+1,j), r_y vertical."""
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


def discrete_laplacian(f, L):
    return (np.roll(f, 1, 0) + np.roll(f, -1, 0)
            + np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f)


def poisson_inverse(rho, L):
    """Solve grad^2 Phi = rho on periodic LxL (zero mean).  Phi = grad^-2 rho."""
    rho_hat = np.fft.fft2(rho)
    kx = 2 * np.pi * np.fft.fftfreq(L)
    ky = 2 * np.pi * np.fft.fftfreq(L)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    # 5-point Laplacian eigenvalues: 2cos kx + 2cos ky - 4
    lap_eig = 2 * np.cos(KX) + 2 * np.cos(KY) - 4
    Phi_hat = np.zeros_like(rho_hat, dtype=complex)
    nz = lap_eig != 0
    Phi_hat[nz] = rho_hat[nz] / lap_eig[nz]
    Phi_hat[0, 0] = 0.0
    Phi = np.real(np.fft.ifft2(Phi_hat))
    return Phi


def edge_sum_field(f, L):
    """Return (rh, rv) moduli from back-reaction r = 1 + alpha*(endpoint sum of f)."""
    rh = 1 + 0.5 * (f + np.roll(f, -1, 0))
    rv = 1 + 0.5 * (f + np.roll(f, -1, 1))
    return rh, rv


def main():
    L = 24
    x = np.arange(L); y = np.arange(L)
    xx, yy = np.meshgrid(x, y)
    cx = cy = L / 2
    rho = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0 ** 2))
    rho = rho - rho.mean()

    # Verify Poisson inverse is correct: grad^2 Phi = rho
    Phi = poisson_inverse(rho, L)
    lap_Phi = discrete_laplacian(Phi, L)
    resid = np.max(np.abs(lap_Phi - rho))
    print("=== Read C: no-preferred-direction on CURVATURE (potential back-reaction) ===")
    print(f"  L={L}, density bump (mean-zero)")
    print(f"  Poisson check: max|grad^2 Phi - rho| = {resid:.2e}  (should be ~0)")

    # ---- Case A/B: r ~ density (contact) ----
    rh, rv = edge_sum_field(rho, L)
    delta_rho = angle_defect_field(rh, rv, L)
    lap_rho = discrete_laplacian(rho, L)
    corr_AB_vs_rho = float(np.corrcoef(delta_rho.ravel(), rho.ravel())[0, 1])
    corr_AB_vs_lap = float(np.corrcoef(delta_rho.ravel(), (-lap_rho).ravel())[0, 1])

    # ---- Case C: r ~ potential (Poisson inverse) ----
    rhC, rvC = edge_sum_field(Phi, L)
    delta_Phi = angle_defect_field(rhC, rvC, L)
    corr_C_vs_rho = float(np.corrcoef(delta_Phi.ravel(), rho.ravel())[0, 1])
    corr_C_vs_lap = float(np.corrcoef(delta_Phi.ravel(), (-lap_rho).ravel())[0, 1])

    print(f"\n  {'back-reaction':>28} | {'corr(delta, rho)':>16} | {'corr(delta, -grad^2 rho)':>22}")
    print(f"  {'A/B: r ~ rho (contact)':>28} | {corr_AB_vs_rho:>16.4f} | {corr_AB_vs_lap:>22.4f}")
    print(f"  {'C:   r ~ Phi (potential)':>28} | {corr_C_vs_rho:>16.4f} | {corr_C_vs_lap:>22.4f}")
    print()
    print("  Read C prediction: case C gives corr(delta, rho) ~ 1 (GR: delta ~ rho),")
    print("  while case A/B gives corr(delta, -grad^2 rho) ~ 1 (contact: delta ~ grad^2 rho).")

    summary = {
        "L": L,
        "poisson_residual": float(resid),
        "AB_r_rho": {"corr_delta_vs_rho": corr_AB_vs_rho, "corr_delta_vs_minus_lap_rho": corr_AB_vs_lap},
        "C_r_Phi": {"corr_delta_vs_rho": corr_C_vs_rho, "corr_delta_vs_minus_lap_rho": corr_C_vs_lap},
        "conclusion": "If case C gives corr(delta, rho) ~ 1, Read C is confirmed: the "
                      "no-preferred-direction acts on curvature (potential back-reaction), "
                      "giving delta ~ rho = GR field equation trace, welding the two halves.",
    }
    out = ROOT / "experiments" / "exp_bridge_B_weld_readC_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
