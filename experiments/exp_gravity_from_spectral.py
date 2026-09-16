"""Gravity from the spectral action itself: defect -> re-minimize S[D] -> modulus
response delta r vs distance.  NO external geometry (no Regge, no hand-placed
r=1+eps rho).

The action is the theory's OWN spectral action (the one that grows the torus in
the structure-selection gate):
    S = -alpha Tr(D^2) + gamma sum(d-c)^2 + delta sum((sum r^4)-c4)^2 + nu F_curv

Base state = toroidal pi-flux (4-regular, equal moduli).  Add a LOCAL defect
(change one edge's modulus), re-minimize S[D], and read how delta r_ij decays
with graph distance from the defect.

Key tension: equal moduli = theorem-1 (no external observer); the defect breaks
that symmetry locally, but the delta-term pulls back to equal moduli.  Whether
delta r ~ 0 (locked) or decays with distance (long-range) is what we measure.

Code: `py -m experiments.exp_gravity_from_spectral`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from itertools import product

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def toroidal_D(n_per_dim, pi_flux=True):
    n = n_per_dim ** 2
    D = np.zeros((n, n), complex)
    for i in range(n_per_dim):
        for j in range(n_per_dim):
            idx = n_per_dim * i + j
            jr = (j + 1) % n_per_dim
            idr = n_per_dim * i + jr
            D[idx, idr] += 1.0
            D[idr, idx] += 1.0
            idd = n_per_dim * ((i + 1) % n_per_dim) + j
            ph = np.pi * j if pi_flux else 0.0
            D[idx, idd] += np.exp(1j * ph)
            D[idd, idx] += np.exp(-1j * ph)
    return D


def D_to_x(D):
    n = D.shape[0]
    ne = n * (n - 1) // 2
    x = np.zeros(ne * 2)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            x[idx] = np.real(D[i, j])
            x[ne + idx] = np.imag(D[i, j])
            idx += 1
    return x


def vec_to_D(x, n):
    ne = n * (n - 1) // 2
    rp = x[:ne]; ip = x[ne:]
    D = np.zeros((n, n), complex)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = rp[idx] + 1j * ip[idx]
            D[j, i] = rp[idx] - 1j * ip[idx]
            idx += 1
    return D


def precompute_cycles(n):
    cycles = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for dd in range(c + 1, n):
                    for cyc in [(a, b, c, dd), (a, b, dd, c), (a, c, b, dd)]:
                        cycles.append(cyc)
    return np.asarray(cycles, dtype=int)


def curvature_term(D, cycles):
    r = np.abs(D)
    p, q, s, t = cycles[:, 0], cycles[:, 1], cycles[:, 2], cycles[:, 3]
    rprod = r[p, q] * r[q, s] * r[s, t] * r[t, p]
    re = np.real(D[p, q] * D[q, s] * D[s, t] * np.conjugate(D[p, t]))
    return float(np.sum(rprod + re))


def make_action(n, alpha, gamma, delta, nu, c, c4, cycles):
    def action(x):
        D = vec_to_D(x, n)
        D2 = D @ D
        tr2 = float(np.real(np.trace(D2)))
        d = np.real(np.diag(D2))
        deg = float(np.sum((d - c) ** 2))
        r = np.abs(D)
        qd = np.sum(r ** 4, axis=1)
        qc = float(np.sum((qd - c4) ** 2))
        curv = curvature_term(D, cycles)
        return -alpha * tr2 + gamma * deg + delta * qc + nu * curv
    return action


def graph_distance(n_per_dim, a, b):
    """Toroidal graph distance between nodes a,b (idx = dim*i + j)."""
    ai, aj = divmod(a, n_per_dim)
    bi, bj = divmod(b, n_per_dim)
    di = min(abs(ai - bi), n_per_dim - abs(ai - bi))
    dj = min(abs(aj - bj), n_per_dim - abs(aj - bj))
    return di + dj


def main():
    n_per_dim = 4
    n = n_per_dim ** 2
    c = 4.0; c4 = 4.0
    alpha, gamma, delta, nu = 2.0, 10.0, 10.0, 1.0
    cycles = precompute_cycles(n)
    fun = make_action(n, alpha, gamma, delta, nu, c, c4, cycles)

    D0 = toroidal_D(n_per_dim, pi_flux=True)
    x0 = D_to_x(D0)
    S0 = fun(x0)
    print("=== gravity from spectral action: defect -> delta r vs distance ===")
    print(f"  n_per_dim={n_per_dim}, N={n}, baseline S = {S0:.4f}")

    # defect: change one edge's modulus near center.
    # center nodes: (i,j)=(1,1) idx=5, (1,2) idx=6 (right edge of (1,1))
    # find the parameter index of edge (5,6)
    ne = n * (n - 1) // 2
    def edge_param_index(i, j):
        # i<j ordering
        a, b = min(i, j), max(i, j)
        # index of (a,b) in upper-triangle
        return a * (2 * n - a - 1) // 2 + (b - a - 1)

    defect_edge = (5, 6)  # right edge of node (1,1), center-ish
    ei = edge_param_index(*defect_edge)
    defect_strength = 1.0  # increase modulus by this much

    x_def = x0.copy()
    x_def[ei] += defect_strength  # bump the real part (modulus) of that edge

    # re-minimize
    res = minimize(fun, x_def, method="L-BFGS-B",
                   options={"maxiter": 8000, "ftol": 1e-12, "gtol": 1e-9})
    D = vec_to_D(res.x, n)
    print(f"  after re-minimize: S = {res.fun:.4f} (baseline {S0:.4f}, delta S = {res.fun - S0:.4f})")

    # delta r per edge, vs graph distance from the defect edge's midpoint
    r0 = np.abs(D0)
    r1 = np.abs(D)
    defect_mid = defect_edge[0]  # distance measured from this node
    print(f"\n  {'graph dist':>10} {'mean |delta r|':>16} {'n_edges':>8}")
    dist_bins = {}
    for i in range(n):
        for j in range(i + 1, n):
            dr = abs(r1[i, j] - r0[i, j])
            d = min(graph_distance(n_per_dim, i, defect_mid),
                    graph_distance(n_per_dim, j, defect_mid))
            dist_bins.setdefault(d, []).append(dr)
    results = []
    for d in sorted(dist_bins.keys()):
        v = float(np.mean(dist_bins[d]))
        results.append({"dist": d, "mean_delta_r": v, "n_edges": len(dist_bins[d])})
        print(f"  {d:>10} {v:>16.4e} {len(dist_bins[d]):>8}")

    summary = {
        "n_per_dim": n_per_dim, "N": n,
        "baseline_S": float(S0), "S_after": float(res.fun),
        "defect_edge": list(defect_edge), "defect_strength": defect_strength,
        "dist_scan": results,
        "conclusion": "delta r vs graph distance from the defect.  If mean|delta r| "
                      "decays with distance => defect produces a long-range modulus "
                      "response (theory grows its own potential).  If ~0 everywhere => "
                      "equal-moduli is locked (delta term pulls back), meaning the "
                      "defect does NOT curve geometry under this action.",
    }
    out = ROOT / "experiments" / "exp_gravity_from_spectral_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
