"""Response function chi(q) = delta K(q) / delta rho(q), the weld discriminator.

Correct structure (per the field-equation framework):
    dS/dr = 0  =>  grad^2 r ~ K   (bond order, via Hellmann-Feynman)
    K related to rho via response:  delta K(q) = chi(q) delta rho(q)
    => grad^2 r ~ chi(q) rho
    chi(q->0) = const  =>  grad^2 r ~ rho  =>  GR (Poisson)
    chi(q->0) ~ q^2    =>  contact

This script MEASURES chi(q) directly (no state choice presupposed):
  1. pi-flux torus, valence band (E<0).
  2. bond-order "force" f_x(i) = dE_gs/dr_{i,i+x} = 2 (-1)^y P[i,i+x] (real),
     and density rho(i) = P[i,i].
  3. perturb with a single-frequency potential V = eps cos(q x).
  4. compute delta f_x, delta rho, project onto cos(q x), ratio = chi(q).
  5. scan q -> small q, read the q->0 limit.

Code: `py -m experiments.exp_bridge_B_response`
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
    L = 32
    N = L * L
    eps = 1e-3
    x = np.arange(L); y = np.arange(L)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    D0 = torus2D(L)
    P0 = valence(D0)
    rho0 = np.real(np.diag(P0)).reshape(L, L)
    f0 = bond_force_x(P0, L)

    print("=== response function chi(q) = delta K(q)/delta rho(q), pi-flux ===")
    print(f"  L={L}, eps={eps}, valence band (E<0)")
    print(f"  {'q (n=2pi n/L)':>14} {'chi(q)':>14} {'reading':>20}")
    results = []
    for n in [1, 2, 3, 4, 6, 8, 12, 16]:
        q = 2 * np.pi * n / L
        cq = np.cos(q * xx)  # single-frequency potential profile
        V = eps * cq.ravel()
        D = D0 + np.diag(V)
        P = valence(D)
        rho = np.real(np.diag(P)).reshape(L, L)
        f = bond_force_x(P, L)
        drho = rho - rho0
        df = f - f0
        # project onto cos(q x)
        denom_r = np.sum(cq * cq)
        proj_drho = np.sum(drho * cq) / denom_r
        proj_df = np.sum(df * cq) / denom_r
        chi = proj_df / proj_drho if abs(proj_drho) > 1e-14 else np.nan
        results.append({"n": n, "q": float(q), "chi": float(chi),
                        "drho_proj": float(proj_drho), "df_proj": float(proj_df)})
        print(f"  {n:>14} {chi:>14.4e}")

    # fit chi(q) ~ q^p in log-log for small q
    qs = np.array([r["q"] for r in results if not np.isnan(r["chi"])])
    chis = np.array([abs(r["chi"]) for r in results if not np.isnan(r["chi"])])
    if len(qs) >= 3 and (chis > 0).all():
        p, _ = np.polyfit(np.log(qs), np.log(chis), 1)
        print(f"\n  log-log chi ~ q^p : p = {p:.3f}")
        print(f"  p~0 => chi->const => GR;  p~2 => contact;  p<0 => chi diverges (unusual)")
        if abs(p) < 0.5:
            verdict = "chi(q->0) ~ const => grad^2 r ~ rho => GR"
        elif abs(p - 2) < 0.7:
            verdict = "chi(q->0) ~ q^2 => contact"
        else:
            verdict = f"chi(q->0) ~ q^{p:.2f} (neither clean const nor q^2)"
    else:
        p = None
        verdict = "insufficient data"

    summary = {
        "L": L, "eps": eps, "scan": results, "loglog_slope_p": p, "verdict": verdict,
        "conclusion": "chi(q)=delta K/delta rho measured directly (no state choice). "
                      "chi(q->0)~const => GR; ~q^2 => contact.  See verdict.",
    }
    out = ROOT / "experiments" / "exp_bridge_B_response_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  verdict: {verdict}")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
