"""Gap C first check: what does the 3D DIRAC propagator actually decay like?

Standard result (massless propagator, spatial Green's function):
    2D scalar  -> log r
    2D Dirac   -> 1/r
    3D scalar  -> 1/r   (Coulomb)
    3D Dirac   -> 1/r^2

So lifting the 2D Dirac (1/r) to 3D gives 1/r^2, NOT 1/r.  The "1/r" (Coulomb/
Newtonian) is the 3D SCALAR (or spin-1/2) propagator, not the 3D Dirac.

This tests the user's plan "2D Dirac -> S^3 -> smooth 1/r": the first half
(stagger disappears) may be right, but the exponent becomes 1/r^2, not 1/r.

Code: `py -m experiments.exp_gravity_gapC`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def dirac_3d_operator(N):
    """3D massless Dirac operator on an N^3 lattice, sigma . sin(k)."""
    # momentum-space Dirac operator: D(k) = sx sin(kx) + sy sin(ky) + sz sin(kz)
    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]], complex)
    sz = np.array([[1, 0], [0, -1]], complex)
    ks = 2 * np.pi * np.arange(N) / N
    # build D as a 2N^3 x 2N^3 matrix (2 spin components per site)
    dim = 2 * N ** 3
    D = np.zeros((dim, dim), complex)
    for idx in range(N ** 3):
        kx_idx = idx // (N * N)
        ky_idx = (idx // N) % N
        kz_idx = idx % N
        kx, ky, kz = ks[kx_idx], ks[ky_idx], ks[kz_idx]
        block = sx * np.sin(kx) + sy * np.sin(ky) + sz * np.sin(kz)
        D[2 * idx:2 * idx + 2, 2 * idx:2 * idx + 2] = block
    return D


def graph_dist_3d(N, a, b):
    ai, aj, ak = a // (N * N), (a // N) % N, a % N
    bi, bj, bk = b // (N * N), (b // N) % N, b % N
    return (min(abs(ai - bi), N - abs(ai - bi))
            + min(abs(aj - bj), N - abs(aj - bj))
            + min(abs(ak - bk), N - abs(ak - bk)))


def main():
    print("=" * 74)
    print("GAP C: 3D Dirac propagator exponent (is it 1/r or 1/r^2?)")
    print("=" * 74)

    N = 8  # 8^3 = 512 sites, 1024 dim
    D = dirac_3d_operator(N)
    eps = 0.05
    G = np.linalg.inv(D + 1j * eps * np.eye(D.shape[0]))

    # |G(r)| vs 3D graph distance, from site 0, spin up -> spin DOWN (off-diagonal)
    dists = {}
    for b in range(N ** 3):
        d = graph_dist_3d(N, 0, b)
        gval = abs(G[0, 2 * b + 1])  # spin-up to spin-down
        dists.setdefault(d, []).append(gval)

    ds = sorted(dists.keys())
    vals = [float(np.mean(dists[d])) for d in ds if d >= 1]
    ds = [d for d in ds if d >= 1]
    print(f"\n  {'dist':>5} {'|G(r)|':>12}")
    for d, v in zip(ds[:12], vals[:12]):
        print(f"  {d:>5} {v:>12.5e}")

    ds_arr = np.array(ds, float); vals_arr = np.array(vals)
    m = (ds_arr >= 2) & (ds_arr <= 6)
    p = -np.polyfit(np.log(ds_arr[m]), np.log(vals_arr[m]), 1)[0]
    print(f"\n  fit |G(r)| ~ r^{{-{p:.3f}}}  (3D Dirac => ~2;  3D scalar => ~1)")

    verdict = ("1/r^2 (3D DIRAC)" if p > 1.5 else ("1/r (3D scalar/Coulomb)" if 0.7 < p < 1.4 else "other"))
    print(f"  VERDICT: {verdict}")
    print(f"\n  => lifting the 2D Dirac (1/r) to 3D gives {verdict}, NOT the 1/r Coulomb.")
    print("     The '1/r' (Newtonian) is the 3D SCALAR/graviton, not the 3D Dirac.")

    out = ROOT / "experiments" / "exp_gravity_gapC_last_run.json"
    out.write_text(json.dumps({"power": float(p), "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
