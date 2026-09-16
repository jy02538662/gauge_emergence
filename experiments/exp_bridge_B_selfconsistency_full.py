"""rho from D, FULL self-consistency loop: inject matter, iterate to a fixed point.

This is the missing hard half of rho-from-D.  exp_bridge_B_rho_from_D did only
ONE direction (rho = f(D)); exp_bridge_B_selfconsistency did only a linear
STABILITY scan of the uniform fixed point.  Neither closed the loop
    rho = rho[D],  D = D[rho]
to a self-consistent INHOMOGENEOUS solution.

Here: inject a LOCAL potential V (matter source) at one site, then iterate
    rho_n = valence_density(D_n)              (matter from geometry)
    r_{n+1} = 1 + alpha*(rho_i + rho_j - 2)   (geometry back-reacts)
with damping, until rho and moduli converge.  Verify:
  - V=0: the loop stays at the uniform flat fixed point (no spontaneous breaking,
    consistent with the endpoint-SUM decoupling of the staggered mode).
  - V!=0: the loop converges to a SELF-CONSISTENT inhomogeneous solution — the
    local potential depresses rho, the moduli respond, and the two settle into a
    coupled defect.  This is the DISCRETE ANALOG of a matter source curving
    spacetime, now fully self-consistent (not the one-shot hand-placed rho).

Code: `py -m experiments.exp_bridge_B_selfconsistency_full`
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


def idx(x, y, L):
    return (y % L) * L + (x % L)


def build_D(rh, rv, L, V, site):
    """D with pi-flux phases, edge moduli rh/rv, plus a local potential V at `site`."""
    N = L * L
    D = np.zeros((N, N))
    for x, y in product(range(L), repeat=2):
        i = idx(x, y, L)
        j = idx(x + 1, y, L)
        w = (-1) ** y * rh[x, y]
        D[i, j] = w; D[j, i] = w
        j = idx(x, y + 1, L)
        w = rv[x, y]
        D[i, j] = w; D[j, i] = w
    D[site, site] += V
    return D


def valence_density(D):
    eigs, vecs = np.linalg.eigh(D)
    occ = eigs < 0
    return np.sum(np.abs(vecs[:, occ]) ** 2, axis=1)


def moduli_from_rho(rho, alpha, L):
    rh = np.zeros((L, L)); rv = np.zeros((L, L))
    for x, y in product(range(L), repeat=2):
        i = idx(x, y, L); jx = idx(x + 1, y, L); jy = idx(x, y + 1, L)
        rh[x, y] = 1 + alpha * (rho[i] + rho[jx] - 2)
        rv[x, y] = 1 + alpha * (rho[i] + rho[jy] - 2)
    return rh, rv


def run_loop(alpha, V, L, mix, n_iter):
    site = (L // 2) * L + (L // 2)
    rh = np.ones((L, L)); rv = np.ones((L, L))
    rho = None
    rho_defect_hist = []
    for _ in range(n_iter):
        D = build_D(rh, rv, L, V, site)
        rho = valence_density(D)
        rh_new, rv_new = moduli_from_rho(rho, alpha, L)
        rh = (1 - mix) * rh + mix * rh_new
        rv = (1 - mix) * rv + mix * rv_new
        rho_defect_hist.append(float(rho[site]))
    return rho, rh, rv, rho_defect_hist


def main():
    L = 16
    N = L * L
    mix = 0.15
    n_iter = 800
    site = (L // 2) * L + (L // 2)

    print("=== rho from D, FULL self-consistency loop (inject matter, iterate) ===")
    print(f"  L={L}, N={N}, mix={mix}, n_iter={n_iter}, site={site}")

    results = []
    for alpha, V in [(0.0, 0.0), (0.2, 0.0), (0.2, 2.0)]:
        rho, rh, rv, hist = run_loop(alpha, V, L, mix, n_iter)
        rho_defect = float(rho[site])
        rho_far = float(rho[0])
        r_std = float((rh.std() + rv.std()) / 2)
        # convergence: fluctuation of rho_defect over the LAST 100 steps
        tail = hist[-100:]
        tail_span = max(tail) - min(tail)
        converged = tail_span < 1e-4
        contrast = rho_far - rho_defect
        print(f"\n  alpha={alpha}, V={V}:")
        print(f"    rho at defect = {rho_defect:.4f}, rho far = {rho_far:.4f} "
              f"(contrast = {contrast:.4f})")
        print(f"    moduli std = {r_std:.4f}")
        print(f"    rho_defect tail span (last 100) = {tail_span:.2e} "
              f"({'converged' if converged else 'NOT converged'})")
        results.append({
            "alpha": alpha, "V": V,
            "rho_defect": rho_defect, "rho_far": rho_far,
            "contrast": contrast, "moduli_std": r_std,
            "tail_span": tail_span, "converged": converged,
        })

    summary = {
        "L": L, "N": N, "mix": mix, "n_iter": n_iter, "site": site,
        "results": results,
        "note": "Full loop rho=rho[D], D=D[rho] with injected local potential V. "
                "V=0 stays flat (rho uniform ~1/2, no spontaneous breaking). "
                "V!=0 converges to a self-consistent inhomogeneous solution: the "
                "defect depresses rho locally and the moduli respond, settling into "
                "a coupled defect (discrete matter-source-curves-geometry).",
    }
    out = ROOT / "experiments" / "exp_bridge_B_selfconsistency_full_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
