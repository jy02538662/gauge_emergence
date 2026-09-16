"""Phase scan: does the metric (edge-modulus) response to a local defect
ever become LONG-RANGE (power law) instead of short-range (exponential)?

Scans flux Phi and modulus-lock coupling delta, measures delta r(r) vs graph
distance after re-minimizing, and fits both exponential and power-law decay.

Long-range = power-law (slow) or divergent correlation length.
Short-range = exponential (fast).

Code: `py -m experiments.exp_gravity_phase_scan`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def toroidal_D(n_per_dim, Phi):
    """Toroidal lattice with uniform flux Phi per plaquette."""
    n = n_per_dim ** 2
    D = np.zeros((n, n), complex)
    for i in range(n_per_dim):
        for j in range(n_per_dim):
            idx = n_per_dim * i + j
            jr = (j + 1) % n_per_dim
            D[idx, n_per_dim * i + jr] += 1.0
            D[n_per_dim * i + jr, idx] += 1.0
            ph = Phi * j
            D[idx, n_per_dim * ((i + 1) % n_per_dim) + j] += np.exp(1j * ph)
            D[n_per_dim * ((i + 1) % n_per_dim) + j, idx] += np.exp(-1j * ph)
    return D


def D_to_x(D):
    n = D.shape[0]; ne = n * (n - 1) // 2
    x = np.zeros(ne * 2); idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            x[idx] = np.real(D[i, j]); x[ne + idx] = np.imag(D[i, j]); idx += 1
    return x


def x_to_D(x, n):
    ne = n * (n - 1) // 2
    rp, ip = x[:ne], x[ne:]
    D = np.zeros((n, n), complex); idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = rp[idx] + 1j * ip[idx]; D[j, i] = rp[idx] - 1j * ip[idx]; idx += 1
    return D


def make_full_action(n_per_dim):
    n = n_per_dim ** 2
    c, c4 = 4.0, 4.0
    cyc = np.asarray([t for comb in combinations(range(n), 4)
                      for t in [comb, (comb[0], comb[1], comb[3], comb[2]),
                                (comb[0], comb[2], comb[1], comb[3])]], dtype=int)

    def full_S(x, alpha, gamma, delta, nu):
        D = x_to_D(x, n); D2 = D @ D
        tr2 = float(np.real(np.trace(D2)))
        d = np.real(np.diag(D2)); deg = float(np.sum((d - c) ** 2))
        r = np.abs(D)
        qd = np.sum(r ** 4, axis=1); qc = float(np.sum((qd - c4) ** 2))
        p, q, s, t = cyc[:, 0], cyc[:, 1], cyc[:, 2], cyc[:, 3]
        rprod = r[p, q] * r[q, s] * r[s, t] * r[t, p]
        re = np.real(D[p, q] * D[q, s] * D[s, t] * np.conjugate(D[p, t]))
        curv = float(np.sum(rprod + re))
        return -alpha * tr2 + gamma * deg + delta * qc + nu * curv
    return full_S


def graph_dist(npd, a, b):
    ai, aj = divmod(a, npd); bi, bj = divmod(b, npd)
    return min(abs(ai - bi), npd - abs(ai - bi)) + min(abs(aj - bj), npd - abs(aj - bj))


def defect_response(npd, Phi, alpha, gamma, delta, nu):
    n = npd ** 2
    full = make_full_action(npd)
    fun = lambda x: full(x, alpha, gamma, delta, nu)
    D0 = toroidal_D(npd, Phi)
    x0 = D_to_x(D0)
    # defect on edge (0,0)-(0,1) => nodes 0 and 1; param index for (0,1)
    # upper-triangle index of (a,b) with a<b
    ne = n * (n - 1) // 2
    a, b = 0, 1
    ei = a * (2 * n - a - 1) // 2 + (b - a - 1)
    xd = x0.copy(); xd[ei] += 1.0
    res = minimize(fun, xd, method="L-BFGS-B",
                   options={"maxiter": 8000, "ftol": 1e-11, "gtol": 1e-8})
    Dr = x_to_D(res.x, n)
    r0 = np.abs(D0); r1 = np.abs(Dr)
    dists = {}
    for i in range(n):
        for j in range(i + 1, n):
            dr = abs(r1[i, j] - r0[i, j])
            d = min(graph_dist(npd, i, 0), graph_dist(npd, j, 0))
            dists.setdefault(d, []).append(dr)
    ds = sorted(dists.keys())
    vals = [float(np.mean(dists[d])) for d in ds]
    return ds, vals


def fit_decay(ds, vals):
    ds = np.asarray(ds, float); vals = np.asarray(vals)
    # exponential: vals ~ A exp(-r/xi)  -> log vals vs r linear
    pos = vals > 1e-15
    if pos.sum() < 2:
        return None
    dsp, vp = ds[pos], np.log(vals[pos])
    xi = -1.0 / np.polyfit(dsp, vp, 1)[0]  # correlation length (positive)
    # power law: vals ~ A r^-p
    p = -np.polyfit(np.log(dsp[ds[pos] > 0]), vp[ds[pos] > 0], 1)[0]
    return {"xi": float(xi), "power": float(p), "d_vals": list(zip(ds.tolist(), vals.tolist()))}


def main():
    npd = 4
    alpha, gamma, nu = 2.0, 10.0, 1.0
    print("=" * 74)
    print(f"PHASE SCAN: defect response decay vs flux Phi and modulus-lock delta")
    print(f"  {npd}x{npd} torus, alpha={alpha} gamma={gamma} nu={nu}")
    print("=" * 74)

    print(f"\n{'Phi/pi':>7} {'delta':>7} {'xi (corr len)':>14} {'power':>8} {'delta r(r=0..)':>28}")
    results = []
    for Phi_over_pi in [0.0, 0.25, 0.5, 1.0]:
        for delta in [10.0, 1.0, 0.1, 0.01]:
            Phi = Phi_over_pi * np.pi
            ds, vals = defect_response(npd, Phi, alpha, gamma, delta, nu)
            fit = fit_decay(ds, vals)
            label = f"{Phi_over_pi:>7} {delta:>7}"
            if fit:
                rstr = " ".join(f"{v:.1e}" for v in vals[:4])
                print(f"{label} {fit['xi']:>14.3f} {fit['power']:>8.3f}  {rstr}")
                results.append({"Phi_over_pi": Phi_over_pi, "delta": delta,
                                "xi": fit["xi"], "power": fit["power"],
                                "vals": fit["d_vals"]})
            else:
                print(f"{label}   (no decay data)")
                results.append({"Phi_over_pi": Phi_over_pi, "delta": delta,
                                "xi": None, "power": None, "vals": []})

    # interpretation
    print("\n--- interpretation ---")
    print("  xi ~ finite (small)   => exponential => SHORT range")
    print("  xi -> large / power ~ small => power-law tail => LONG range")
    out = ROOT / "experiments" / "exp_gravity_phase_scan_last_run.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
