"""Attack the coefficient alpha (8 pi G analog): first, grow rho from D itself.

The back-reaction law r_ij - 1 = alpha (rho_i + rho_j - 2) has a coefficient alpha
whose physical status is entangled with rho's normalization, the mass m, and the
scale Lambda.  But in the current chain rho is HAND-PLACED (a Gaussian bump in
exp_bridge_B_matter_curvature).  Before alpha's meaning can be fixed, rho must
GROW FROM D: rho = fermion density = sum over occupied (E<0) states of |psi_n|^2.

This script does that:
  1. 2D pi-flux D (L x L).  Diagonalize, fill the valence band (E < 0), compute
     the ground-state density rho_i = sum_{E_n<0} |psi_n(i)|^2.
  2. No matter (pure pi-flux): rho is UNIFORM (= 1/2 per site) => flat => no
     curvature (this is "flat = no matter" grown from D, not hand-placed).
  3. Staggered mass m: gap opens; rho still translation-invariant (staggered).
  4. Local potential V at one site: rho responds locally => genuine inhomogeneous
     density => the real source of curvature (replaces the hand-placed bump).

This gives the rho side of the self-consistency loop  rho = rho[D], D = D[rho],
which is where alpha's status gets fixed.

Code: `py -m experiments.exp_bridge_B_rho_from_D`
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


def torus2D(L, flux=True):
    N = L * L

    def idx(x, y):
        return (y % L) * L + (x % L)

    D = np.zeros((N, N))
    for x, y in product(range(L), repeat=2):
        i = idx(x, y)
        j = idx(x + 1, y); w = (-1) ** y if flux else 1.0; D[i, j] = w; D[j, i] = w
        j = idx(x, y + 1); w = 1.0; D[i, j] = w; D[j, i] = w
    return D


def valence_density(D):
    """rho_i = sum over occupied (E<0) states |psi_n(i)|^2."""
    eigs, vecs = np.linalg.eigh(D)
    occ = eigs < 0
    rho = np.sum(np.abs(vecs[:, occ]) ** 2, axis=1)
    return rho, eigs, occ


def main():
    L = 12
    N = L * L
    x = np.arange(L); y = np.arange(L)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    stagger = ((-1) ** (xx + yy)).ravel()  # staggered +/-1 for mass

    # ---- 1. no matter: pure pi-flux => rho uniform ----
    D0 = torus2D(L, flux=True)
    rho0, eigs0, occ0 = valence_density(D0)
    n_occ0 = int(occ0.sum())

    # ---- 2. staggered mass m => gap opens, rho staggered ----
    m = 0.5
    Dm = D0 + m * np.diag(stagger)
    rho_m, eigs_m, occ_m = valence_density(Dm)

    # ---- 3. local potential V at one site => rho responds locally ----
    V = 2.0
    site = N // 2  # central site
    DV = D0.copy()
    DV[site, site] += V
    rho_V, eigs_V, occ_V = valence_density(DV)

    print("=== rho grows from D: valence-band density (the real matter source) ===")
    print(f"  L={L}, N={N}, n_occupied (E<0) = {n_occ0}")
    print(f"  1. pure pi-flux:  rho std = {rho0.std():.3e},  min={rho0.min():.4f}, "
          f"max={rho0.max():.4f}  (uniform ~1/2 => flat)")
    print(f"  2. staggered mass m={m}: rho std = {rho_m.std():.3e}  "
          f"(staggered: even/odd differ, still translation-invariant)")
    print(f"     even-site rho = {rho_m[0::2].mean():.4f}, odd-site rho = {rho_m[1::2].mean():.4f}")
    print(f"  3. local potential V={V} at site {site}: rho std = {rho_V.std():.3e}")
    print(f"     rho at defect = {rho_V[site]:.4f}, rho far = {rho_V.mean():.4f} "
          f"(local response => inhomogeneous density)")

    summary = {
        "L": L, "N": N, "n_occupied": n_occ0,
        "pure_piflux_rho_std": float(rho0.std()),
        "pure_piflux_rho_min": float(rho0.min()),
        "pure_piflux_rho_max": float(rho0.max()),
        "staggered_mass_m": m,
        "staggered_rho_std": float(rho_m.std()),
        "staggered_even_rho": float(rho_m[0::2].mean()),
        "staggered_odd_rho": float(rho_m[1::2].mean()),
        "local_potential_V": V,
        "local_rho_std": float(rho_V.std()),
        "local_rho_at_defect": float(rho_V[site]),
        "local_rho_mean": float(rho_V.mean()),
        "conclusion": "rho grows from D's valence band (E<0). Pure pi-flux => rho uniform "
                      "(=1/2, flat). Staggered mass => staggered rho (translation-invariant). "
                      "Local potential => local rho response (genuine inhomogeneity). This is "
                      "the rho[D] side of the self-consistency loop rho=rho[D], D=D[rho], which "
                      "is where the coupling alpha (8πG analog) gets its status fixed.",
    }
    out = ROOT / "experiments" / "exp_bridge_B_rho_from_D_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
