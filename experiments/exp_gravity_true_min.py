"""Find the TRUE ground state of the full action (re-minimize from pi-flux),
then compute the metric-fluctuation mass (lowest Hessian eigenvalue) THERE.

This is the honest test of long-range vs short-range: the mass^2 of the metric
(edge modulus) fluctuation at the actual minimum decides exponential vs power-law.
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


def toroidal_D(n_per_dim, pi_flux=True):
    n = n_per_dim ** 2
    D = np.zeros((n, n), complex)
    for i in range(n_per_dim):
        for j in range(n_per_dim):
            idx = n_per_dim * i + j
            jr = (j + 1) % n_per_dim
            D[idx, n_per_dim * i + jr] += 1.0
            D[n_per_dim * i + jr, idx] += 1.0
            ph = np.pi * j if pi_flux else 0.0
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


def basis_matrices(n):
    ne = n * (n - 1) // 2
    Ms = []
    for i in range(n):
        for j in range(i + 1, n):
            M = np.zeros((n, n), complex); M[i, j] = 1.0; M[j, i] = 1.0; Ms.append(M)
    for i in range(n):
        for j in range(i + 1, n):
            M = np.zeros((n, n), complex); M[i, j] = 1j; M[j, i] = -1j; Ms.append(M)
    return np.asarray(Ms)


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


def finite_diff_hessian(fun, x0, eps=1e-5):
    n = len(x0); H = np.zeros((n, n)); f0 = fun(x0)
    for a in range(n):
        xp = x0.copy(); xp[a] += eps; xm = x0.copy(); xm[a] -= eps
        H[a, a] = (fun(xp) - 2 * f0 + fun(xm)) / (eps * eps)
    for a in range(n):
        for b in range(a + 1, n):
            xpp = x0.copy(); xpp[a] += eps; xpp[b] += eps
            xpm = x0.copy(); xpm[a] += eps; xpm[b] -= eps
            xmp = x0.copy(); xmp[a] -= eps; xmp[b] += eps
            xmm = x0.copy(); xmm[a] -= eps; xmm[b] -= eps
            H[a, b] = H[b, a] = (fun(xpp) - fun(xpm) - fun(xmp) + fun(xmm)) / (4 * eps * eps)
    return H


def main():
    npd = 3
    n = npd ** 2
    D0 = toroidal_D(npd, pi_flux=True)
    x0 = D_to_x(D0)
    full = make_full_action(npd)
    alpha, gamma, delta, nu = 2.0, 10.0, 10.0, 1.0
    fun = lambda x: full(x, alpha, gamma, delta, nu)

    S0 = fun(x0)
    print(f"N={n} ({npd}x{npd}), pi-flux baseline S = {S0:.6f}")

    res = minimize(fun, x0, method="L-BFGS-B",
                   options={"maxiter": 20000, "ftol": 1e-12, "gtol": 1e-9})
    xmin = res.x
    print(f"after re-minimize: S = {res.fun:.6f}  (delta S = {res.fun - S0:.6f})")
    print(f"  |x - x0| = {np.linalg.norm(xmin - x0):.4f}  (how far from pi-flux)")

    # Hessian at the true minimum
    H = finite_diff_hessian(fun, xmin)
    w = np.linalg.eigvalsh(H)
    print(f"\nHessian at TRUE minimum: {len(w)} eigenvalues")
    print(f"  lowest 8 : {np.round(w[:8], 4)}")
    print(f"  #neg={int(np.sum(w < -1e-6))}, #~zero(<1e-6)={int(np.sum(np.abs(w) < 1e-6))}")

    # metric mass = lowest POSITIVE eigenvalue (the gap of the metric fluctuation)
    pos = w[w > 1e-6]
    print(f"  metric mass^2 (lowest positive eig) = {pos[0]:.4e}")

    # also: response of a local defect at the true minimum (does delta r decay?)
    # quick: bump one edge, re-minimize, measure delta r vs distance
    Dm = x_to_D(xmin, n)
    ne = n * (n - 1) // 2
    def edge_idx(i, j):
        a, b = min(i, j), max(i, j)
        return a * (2 * n - a - 1) // 2 + (b - a - 1)
    # center edge (1,1)-(1,2) -> idx 5,6 for 3x3? use (0,0)-(0,1) edge 0-1
    ei = edge_idx(1, 2)
    xd = xmin.copy(); xd[ei] += 1.0
    resd = minimize(fun, xd, method="L-BFGS-B",
                    options={"maxiter": 20000, "ftol": 1e-12, "gtol": 1e-9})
    Dd = x_to_D(resd.x, n)
    r0 = np.abs(Dm); r1 = np.abs(Dd)
    print(f"\ndefect response (delta |r| vs graph distance from node 1):")
    for i in range(n):
        for j in range(i + 1, n):
            dr = abs(r1[i, j] - r0[i, j])
            dist = min(abs(i - 1), n - abs(i - 1)) + min(abs(j - 1), n - abs(j - 1)) if False else 0
    # simple: report per-edge delta r
    deltas = []
    for i in range(n):
        for j in range(i + 1, n):
            deltas.append((abs(r1[i, j] - r0[i, j]), i, j))
    deltas.sort(reverse=True)
    print("  top 5 edge delta r:", [f"{d[0]:.2e}@({d[1]},{d[2]})" for d in deltas[:5]])

    out = ROOT / "experiments" / "exp_gravity_true_min_last_run.json"
    out.write_text(json.dumps({
        "baseline_S": S0, "min_S": float(res.fun), "delta_S": float(res.fun - S0),
        "dist_from_piflux": float(np.linalg.norm(xmin - x0)),
        "hessian_lowest": w[:8].tolist(),
        "metric_mass2": float(pos[0]),
        "n_neg": int(np.sum(w < -1e-6)), "n_zero": int(np.sum(np.abs(w) < 1e-6)),
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
