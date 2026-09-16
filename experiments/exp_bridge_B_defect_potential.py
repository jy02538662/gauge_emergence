"""Defect -> long-range potential -> curvature concentrated at source.

Correct GR weak-field chain (the one we settled on):
    rho = delta (point defect) => grad^2 Phi = rho => Phi ~ ln r (2D, LONG-RANGE)
    => curvature R = grad^2 Phi ~ rho = delta (CONCENTRATED at source, ~0 far).

Discrete model (Q1 solved): potential = moduli r, curvature = angle defect
delta = cross second derivative of r.

This script tests BOTH facts directly (no "delta ~ rho vs grad^2 rho" ambiguity):
  1. point source rho = delta_i0 - 1/N (mean-zero, Poisson solvability).
  2. solve grad^2 Phi = rho (FFT); verify Phi ~ ln r (long-range: decays SLOWLY).
  3. set moduli r = 1 + alpha*Phi (isotropic); compute angle defect delta.
  4. verify delta is CONCENTRATED at the defect (peak at source, ~0 far away).

Code: `py -m experiments.exp_bridge_B_defect_potential`
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


def poisson_inverse(rho, L):
    """Solve grad^2 Phi = rho on periodic LxL (zero mean)."""
    rho_hat = np.fft.fft2(rho)
    kx = 2 * np.pi * np.fft.fftfreq(L)
    ky = 2 * np.pi * np.fft.fftfreq(L)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    lap_eig = 2 * np.cos(KX) + 2 * np.cos(KY) - 4
    Phi_hat = np.zeros_like(rho_hat, dtype=complex)
    nz = lap_eig != 0
    Phi_hat[nz] = rho_hat[nz] / lap_eig[nz]
    Phi_hat[0, 0] = 0.0
    return np.real(np.fft.ifft2(Phi_hat))


def discrete_laplacian(f, L):
    return (np.roll(f, 1, 0) + np.roll(f, -1, 0)
            + np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f)


def main():
    L = 64
    alpha = 0.3
    x = np.arange(L); y = np.arange(L)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    cx = cy = L // 2

    # 1. point source (defect) at center, mean-zero
    rho = np.zeros((L, L))
    rho[cx, cy] = 1.0
    rho = rho - rho.mean()

    # 2. solve grad^2 Phi = rho
    Phi = poisson_inverse(rho, L)
    resid = np.max(np.abs(discrete_laplacian(Phi, L) - rho))
    print("=== defect -> long-range potential -> curvature at source ===")
    print(f"  L={L}, alpha={alpha}, point defect at center")
    print(f"  Poisson residual max|grad^2 Phi - rho| = {resid:.2e}")

    # verify Phi ~ ln r: sample Phi along a line from center, compare to ln r
    print("\n  1. potential Phi(r) along x from center (should ~ ln r, SLOW decay):")
    # normalize Phi so Phi(center)=0 (gauge), measure |Phi| at distances
    Phi0 = Phi[cx, cy]
    for r in [0, 1, 2, 4, 8, 16, 24, 31]:
        # average over the 4 points at distance r (avoid wrap ambiguity)
        pts = [(cx+r, cy), (cx-r, cy), (cx, cy+r), (cx, cy-r)]
        vals = [Phi[px % L, py % L] for px, py in pts if 0 <= px % L < L and 0 <= py % L < L]
        if vals:
            mean_v = np.mean(vals)
            print(f"    r={r:2d}: Phi(r) = {mean_v:+.4f}  (relative to center {Phi0:+.4f})")
    # fit: Phi(r) ~ ln r => dPhi between r=4 and r=16 should be ~ ln(16/4)=1.386
    dPhi_4_16 = (Phi[cx+16, cy] if cx+16 < L else 0) - (Phi[cx+4, cy] if cx+4 < L else 0)
    print(f"    check: Phi(r=16)-Phi(r=4) = {dPhi_4_16:+.4f}  (ln(16/4)=1.386 for pure ln r)")

    # 3. moduli r = 1 + alpha*Phi (isotropic), curvature delta
    r_x = 1 + alpha * Phi
    r_y = 1 + alpha * Phi
    delta = angle_defect_field(r_x, r_y, L)

    # 4. is delta concentrated at the source?
    print("\n  2. curvature delta (angle defect): peak at source, ~0 far?")
    print(f"    delta at defect = {delta[cx, cy]:+.4e}")
    # max |delta| location
    dmax = np.max(np.abs(delta))
    print(f"    max|delta| = {dmax:.3e}")
    # value at a few distances
    for r in [0, 1, 2, 4, 8, 16]:
        pts = [(cx+r, cy), (cx-r, cy), (cx, cy+r), (cx, cy-r)]
        vals = [delta[px % L, py % L] for px, py in pts]
        print(f"    r={r:2d}: |delta| = {np.mean(np.abs(vals)):+.4e}")

    summary = {
        "L": L, "alpha": alpha,
        "poisson_residual": float(resid),
        "Phi_center": float(Phi0),
        "dPhi_4_16": float(dPhi_4_16),
        "delta_at_defect": float(delta[cx, cy]),
        "delta_max": float(dmax),
        "delta_far_r16": float(np.mean(np.abs([delta[(cx+16) % L, cy],
                                                delta[(cx-16) % L, cy],
                                                delta[cx, (cy+16) % L],
                                                delta[cx, (cy-16) % L]]))),
        "conclusion": "Check: Phi ~ ln r (long-range, SLOW decay), delta concentrated "
                      "at defect (peak at source, decays fast / ~0 far).  If both hold, "
                      "the chain 'defect -> long-range potential -> curvature at source' "
                      "is confirmed in the discrete model.",
    }
    out = ROOT / "experiments" / "exp_bridge_B_defect_potential_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
