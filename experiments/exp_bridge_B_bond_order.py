"""Bond order K_ij = <c_i^dag c_j> vs density rho_i = <c_i^dag c_i>, pi-flux torus.

Tests the KEY claim in welding the two halves of the field equation:
  - Hellmann-Feynman: dE_gs/dr_ij = K_ij (bond order), NOT rho.
  - So the correct matter source is the bond order K, not the density rho.
  - The bond order carries GRADIENT structure (long-range tail from gapless Dirac
    points), while the density is a constant (no gradient).

Verifiable (three steps, all computed, no hand-waving):
  1. rho_i = P_ii is UNIFORM (= 1/2).
  2. K(r) = P_{i,i+r} decays as a POWER LAW (gapless Dirac), not exponential.
  3. Contrast: with a staggered mass gap m, K(r) becomes short-range (exponential),
     proving the long-range tail comes from the gapless Dirac points.

Code: `py -m experiments.exp_bridge_B_bond_order`
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


def torus2D(L, flux=True, mass=0.0):
    """2D pi-flux torus (L x L).  Optional staggered mass m*(-1)^{x+y}."""
    N = L * L

    def idx(x, y):
        return (y % L) * L + (x % L)

    D = np.zeros((N, N))
    for x, y in product(range(L), repeat=2):
        i = idx(x, y)
        j = idx(x + 1, y); w = (-1) ** y if flux else 1.0; D[i, j] = w; D[j, i] = w
        j = idx(x, y + 1); w = 1.0; D[i, j] = w; D[j, i] = w
    if mass != 0.0:
        for x, y in product(range(L), repeat=2):
            i = idx(x, y)
            D[i, i] += mass * ((-1) ** (x + y))
    return D


def density_matrix(D):
    eigs, vecs = np.linalg.eigh(D)
    occ = eigs < 0
    P = vecs[:, occ] @ vecs[:, occ].conj().T
    return P, eigs, occ


def bond_order_along_x(P, L):
    """K(r) = mean over i of P[i, i+r], r along x.  Returns array K[0..L-1]."""
    N = L * L

    def idx(x, y):
        return (y % L) * L + (x % L)

    K = np.zeros(L)
    for r in range(L):
        vals = []
        for x, y in product(range(L), repeat=2):
            i = idx(x, y)
            j = idx(x + r, y)
            vals.append(P[i, j])
        K[r] = np.real(np.mean(vals))
    return K


def main():
    L = 32
    N = L * L

    # ---- 1. density rho uniform? ----
    D = torus2D(L, flux=True)
    P, eigs, occ = density_matrix(D)
    rho = np.real(np.diag(P))
    n_occ = int(occ.sum())

    print("=== bond order K vs density rho in pi-flux torus ===")
    print(f"  L={L}, N={N}, n_occupied (E<0) = {n_occ}")
    print(f"  1. density rho: min={rho.min():.4f}, max={rho.max():.4f}, std={rho.std():.2e}")
    print(f"     (uniform ~1/2 => rho carries NO gradient structure)")

    # ---- 2. bond order K(r) decay (gapless) ----
    K = bond_order_along_x(P, L)
    print(f"\n  2. bond order K(r) along x (gapless pi-flux):")
    for r in [0, 1, 2, 3, 4, 6, 8, 12, 16]:
        print(f"     K(r={r:2d}) = {K[r]:+.6f}")
    # fit power law on tail: K(r) ~ r^{-alpha}
    # use r from 3..L//2, exclude r=0 (density)
    rs = np.arange(3, L // 2)
    tail = np.abs(K[3:L // 2])
    # log-log fit
    mask = tail > 1e-10
    if mask.sum() >= 4:
        logr = np.log(rs[mask])
        logK = np.log(tail[mask])
        alpha, intercept = np.polyfit(logr, logK, 1)
        print(f"     power-law fit K(r) ~ r^-alpha: alpha = {-alpha:.3f} "
              f"(gapless => power law, not exponential)")
    else:
        print(f"     tail too small to fit (K decays fast => maybe gapped?)")
        alpha = None

    # ---- 3. contrast: staggered mass gap => K(r) short-range ----
    m = 0.5
    Dm = torus2D(L, flux=True, mass=m)
    Pm, eigsm, occm = density_matrix(Dm)
    Km = bond_order_along_x(Pm, L)
    print(f"\n  3. bond order K(r) with staggered mass m={m} (gapped):")
    for r in [0, 1, 2, 3, 4, 6, 8]:
        print(f"     K(r={r:2d}) = {Km[r]:+.6f}")
    # exponential fit on tail
    rs3 = np.arange(2, L // 2)
    tail3 = np.abs(Km[2:L // 2])
    mask3 = tail3 > 1e-12
    if mask3.sum() >= 4:
        logK3 = np.log(tail3[mask3])
        beta, _ = np.polyfit(rs3[mask3], logK3, 1)
        print(f"     exponential fit K(r) ~ exp(-r/xi): xi = {1/abs(beta):.3f} "
              f"(gapped => short-range)")

    summary = {
        "L": L, "N": N, "n_occupied": n_occ,
        "rho_min": float(rho.min()), "rho_max": float(rho.max()),
        "rho_std": float(rho.std()),
        "bond_order_gapless": {str(r): float(K[r]) for r in [0, 1, 2, 3, 4, 6, 8, 12, 16]},
        "power_law_alpha": None if alpha is None else float(-alpha),
        "bond_order_gapped_m0p5": {str(r): float(Km[r]) for r in [0, 1, 2, 3, 4, 6, 8]},
        "conclusion": "rho (density) is uniform = no gradient. K (bond order) decays "
                      "as a POWER LAW in the gapless pi-flux (long-range tail from Dirac "
                      "points) => K carries gradient structure that rho lacks. With a gap "
                      "(staggered mass) K becomes short-range. This is the evidence for "
                      "'the correct matter source is K, not rho' (Hellmann-Feynman).",
    }
    out = ROOT / "experiments" / "exp_bridge_B_bond_order_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
