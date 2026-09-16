"""Modulus defect vs PHASE defect: which response is long-range?

User's physical guess (sharp):
  - modulus = metric (gapped, SHORT-range)
  - phase = gauge field (topological holonomy, possibly MASSLESS -> LONG-range)

Test: on the pi-flux torus, add a defect that perturbs (a) one edge's MODULUS
vs (b) one edge's PHASE, re-minimize S[D], and compare how the response decays
with graph distance.

Prediction: modulus response decays fast (exponential, gapped), phase response
decays slow (power-law / flat, massless gauge).  If confirmed, the theory's own
"matter -> geometry" is a SHORT-range (contact) + LONG-range (gauge/gravity)
structure, without borrowing any external geometry.

Code: `py -m experiments.exp_gravity_phase`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
    ai, aj = divmod(a, n_per_dim)
    bi, bj = divmod(b, n_per_dim)
    di = min(abs(ai - bi), n_per_dim - abs(ai - bi))
    dj = min(abs(aj - bj), n_per_dim - abs(aj - bj))
    return di + dj


def edge_param_index(n, i, j):
    a, b = min(i, j), max(i, j)
    return a * (2 * n - a - 1) // 2 + (b - a - 1)


def run_defect(n_per_dim, defect_edge, defect_kind, strength, fun, n, cycles):
    D0 = toroidal_D(n_per_dim, pi_flux=True)
    x0 = D_to_x(D0)
    ne = n * (n - 1) // 2
    ei = edge_param_index(n, *defect_edge)
    x_def = x0.copy()
    if defect_kind == "modulus":
        x_def[ei] += strength          # bump real part (modulus)
    else:  # phase
        x_def[ne + ei] += strength     # bump imaginary part (phase)
    res = minimize(fun, x_def, method="L-BFGS-B",
                   options={"maxiter": 8000, "ftol": 1e-12, "gtol": 1e-9})
    D = vec_to_D(res.x, n)
    return D0, D, res.fun


def main():
    n_per_dim = 4
    n = n_per_dim ** 2
    c = 4.0; c4 = 4.0
    alpha, gamma, delta, nu = 2.0, 10.0, 10.0, 1.0
    cycles = precompute_cycles(n)
    fun = make_action(n, alpha, gamma, delta, nu, c, c4, cycles)

    defect_edge = (5, 6)
    defect_mid = defect_edge[0]

    print("=== modulus defect vs phase defect: response decay ===")
    print(f"  N={n}, defect edge {defect_edge}")

    for kind, strength, label in [("modulus", 1.0, "modulus (metric)"),
                                  ("phase", 0.3, "phase (gauge)")]:
        D0, D, S = run_defect(n_per_dim, defect_edge, kind, strength, fun, n, cycles)
        r0, r1 = np.abs(D0), np.abs(D)
        # response: modulus response = |r1-r0|, phase response = |arg change| (imag part)
        dist_bins_mod = {}
        dist_bins_pha = {}
        for i in range(n):
            for j in range(i + 1, n):
                dr = abs(r1[i, j] - r0[i, j])
                dph = abs(np.imag(D[i, j]) - np.imag(D0[i, j]))
                d = min(graph_distance(n_per_dim, i, defect_mid),
                        graph_distance(n_per_dim, j, defect_mid))
                dist_bins_mod.setdefault(d, []).append(dr)
                dist_bins_pha.setdefault(d, []).append(dph)
        print(f"\n  [{label}] defect strength={strength}:")
        print(f"    {'dist':>5} {'modulus resp':>14} {'phase resp':>14}")
        rows = []
        for d in sorted(dist_bins_mod.keys()):
            mv = float(np.mean(dist_bins_mod[d]))
            pv = float(np.mean(dist_bins_pha[d]))
            rows.append({"dist": d, "modulus_resp": mv, "phase_resp": pv})
            print(f"    {d:>5} {mv:>14.4e} {pv:>14.4e}")

    summary = {
        "n_per_dim": n_per_dim, "N": n, "defect_edge": list(defect_edge),
        "note": "Modulus response (metric) vs phase response (gauge) decay with distance. "
                "Short-range (gapped metric) vs long-range (massless gauge) is the question.",
    }
    out = ROOT / "experiments" / "exp_gravity_phase_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
