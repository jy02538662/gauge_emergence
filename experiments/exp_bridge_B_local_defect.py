"""Local defect: bond-order response delta K vs distance (power law vs exponential).

The weld source is the bond order K (Hellmann-Feynman theorem), not the density.
We found chi(q->0)=0 (long-wavelength: vacuum incompressible).  The matter is a
LOCAL defect, so the right question is: how does a single-point potential's
bond-order response delta K decay with distance?

  - delta K ~ power law (1/r, 1/r^2, 1/r^3) => defect naturally produces
    long-range bond-order response => matter source is long-range => chain works.
  - delta K ~ exponential => defect response short-range => potential's long
    range needs another mechanism.

Code: `py -m experiments.exp_bridge_B_local_defect`
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


def torus2D(L):
    N = L * L

    def idx(x, y):
        return (y % L) * L + (x % L)

    D = np.zeros((N, N))
    for x, y in product(range(L), repeat=2):
        i = idx(x, y)
        j = idx(x + 1, y); w = (-1) ** y; D[i, j] = w; D[j, i] = w
        j = idx(x, y + 1); w = 1.0; D[i, j] = w; D[j, i] = w
    return D


def valence(D):
    eigs, vecs = np.linalg.eigh(D)
    occ = eigs < 0
    P = vecs[:, occ] @ vecs[:, occ].conj().T
    return P


def bond_force_x(P, L):
    """f_x(x,y) = 2 (-1)^y P[(x,y),(x+1,y)] (real, Hermitian)."""
    N = L * L

    def idx(x, y):
        return (y % L) * L + (x % L)

    f = np.zeros((L, L))
    for x, y in product(range(L), repeat=2):
        i = idx(x, y); j = idx(x + 1, y)
        f[x, y] = 2 * (-1) ** y * np.real(P[i, j])
    return f


def main():
    L = 48
    V0 = 2.0
    N = L * L
    x = np.arange(L); y = np.arange(L)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    cx = cy = L // 2

    D0 = torus2D(L)
    P0 = valence(D0)
    f0 = bond_force_x(P0, L)

    # local defect at center
    D = D0.copy()
    D[cx * L + cy, cx * L + cy] += V0
    P = valence(D)
    f = bond_force_x(P, L)
    df = f - f0

    print("=== local defect: bond-order response delta K vs distance ===")
    print(f"  L={L}, V0={V0} at center ({cx},{cy})")
    print(f"  delta K(r) = |bond_force_x(P)-bond_force_x(P0)| vs distance")
    print(f"  {'r':>3} {'max|deltaK|':>14}")

    # measure |delta f| at Chebyshev distance r, averaged over ring
    # (use max over ring to avoid cancellation)
    vals_by_r = {}
    for px in range(L):
        for py in range(L):
            r = max(abs(px - cx), abs(py - cy))
            vals_by_r.setdefault(r, []).append(abs(df[px, py]))

    results = []
    for r in sorted(vals_by_r.keys()):
        if r > L // 2:
            break
        v = np.max(vals_by_r[r])
        results.append({"r": int(r), "max_deltaK": float(v)})
        if r in [0, 1, 2, 3, 4, 6, 8, 12, 16]:
            print(f"  {r:>3} {v:>14.3e}")

    # fit power law on tail (r=3..L/4): deltaK ~ r^-alpha
    rs = np.array([d["r"] for d in results if d["r"] >= 3 and d["r"] <= L // 4])
    dks = np.array([d["max_deltaK"] for d in results if d["r"] >= 3 and d["r"] <= L // 4])
    mask = dks > 1e-14
    alpha = None
    if mask.sum() >= 4:
        logr = np.log(rs[mask]); logd = np.log(dks[mask])
        alpha, _ = np.polyfit(logr, logd, 1)
        alpha = -alpha
        print(f"\n  power-law fit deltaK ~ r^-alpha (r=3..L/4): alpha = {alpha:.3f}")
        if alpha < 2.5:
            verdict = f"deltaK ~ r^-{alpha:.2f}: long-range (alpha<3), matter source long-range"
        else:
            verdict = f"deltaK ~ r^-{alpha:.2f}: decays fast (alpha>=3), near-local"
    else:
        verdict = "tail too small to fit"

    summary = {
        "L": L, "V0": V0,
        "scan": results,
        "power_law_alpha": alpha,
        "verdict": verdict,
        "conclusion": "delta K vs distance. Power-law exponent alpha decides whether the "
                      "defect produces a long-range bond-order response (matter source). "
                      "alpha=1 => 1/r (like 3D Poisson), alpha=2 => 1/r^2, alpha=3 => 1/r^3.",
    }
    out = ROOT / "experiments" / "exp_bridge_B_local_defect_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  verdict: {verdict}")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
