"""Mass gap (Hessian spectrum) of S[D] at the pi-flux toroidal D0.

Simple action  S = -alpha Tr(D^2) + beta Tr(D^4):
  exact analytic Hessian (no D0^2=4I assumption), cross-checked vs finite-diff.

Full action (degree + modulus-lock + curvature): finite-difference Hessian.

The lowest Hessian eigenvalue = mass^2 of the metric (edge-modulus) fluctuation:
  > 0  => gapped (exponential / short range)
  = 0  => flat direction (massless mode / long-range candidate)
  < 0  => saddle (pi-flux not a stable minimum of that action)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from itertools import combinations

import numpy as np

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


def simple_S(x, alpha, beta):
    n = int(round((1 + np.sqrt(1 + 8 * (len(x) // 2))) / 2))
    D = x_to_D(x, n); D2 = D @ D
    return -alpha * float(np.real(np.trace(D2))) + beta * float(np.real(np.trace(D2 @ D2)))


def simple_hessian(D0, alpha, beta):
    n = D0.shape[0]; D02 = D0 @ D0
    Ms = basis_matrices(n)
    LMs = (-alpha * Ms
           + 4 * beta * np.einsum('ik,akj->aij', D02, Ms)
           + 2 * beta * np.einsum('ik,akl,lj->aij', D0, Ms, D0))
    Q = np.real(np.einsum('aij,bij->ab', Ms.conj(), LMs))
    return 2.0 * Q


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
    print("=" * 74)
    print("HESSIAN SPECTRUM of S[D] at pi-flux  (mass^2 of metric fluctuation)")
    print("=" * 74)

    D0 = toroidal_D(4, pi_flux=True)
    lam = np.linalg.eigvalsh(D0)
    print(f"\nD0 (4x4 pi-flux): spectrum [{lam[0]:.4f}, {lam[-1]:.4f}], "
          f"max|D0^2-4I|={np.max(np.abs(D0@D0-4*np.eye(16))):.3f}")

    print("\n--- (A) simple  -alpha Tr(D^2)+beta Tr(D^4), beta=1 ---")
    print(f"{'alpha':>7} {'lowest':>12} {'highest':>12} {'#neg':>6} {'#~zero':>7}")
    rows = []
    for alpha in [2, 4, 6, 8, 10, 12, 16, 24, 32, 40]:
        H = simple_hessian(D0, alpha, 1.0)
        w = np.linalg.eigvalsh(H)
        nneg = int(np.sum(w < -1e-8)); nz = int(np.sum(np.abs(w) < 1e-8))
        rows.append({"alpha": alpha, "lowest": float(w[0]), "highest": float(w[-1]),
                     "n_neg": nneg, "n_zero": nz})
        print(f"{alpha:>7} {w[0]:>12.4e} {w[-1]:>12.4e} {nneg:>6} {nz:>7}")

    print("\n--- cross-check simple action analytic vs finite-diff (3x3, alpha=8) ---")
    D3 = toroidal_D(3, pi_flux=True)
    w3a = np.sort(np.linalg.eigvalsh(simple_hessian(D3, 8.0, 1.0)))
    w3f = np.sort(np.linalg.eigvalsh(finite_diff_hessian(lambda x: simple_S(x, 8.0, 1.0), D_to_x(D3))))
    print(f"  analytic lowest 5 : {np.round(w3a[:5], 4)}")
    print(f"  finitediff lowest 5: {np.round(w3f[:5], 4)}")
    print(f"  max abs diff (all 72): {np.max(np.abs(w3a - w3f)):.2e}")

    print("\n--- (B) full action (3x3): alpha=2 gamma=10 delta=10 nu=1 ---")
    full = make_full_action(3)
    fun = lambda x: full(x, 2.0, 10.0, 10.0, 1.0)
    Hb = finite_diff_hessian(fun, D_to_x(D3))
    wb = np.linalg.eigvalsh(Hb)
    print(f"  lowest {wb[0]:.4e}, 2nd {wb[1]:.4e}, #neg={int(np.sum(wb<-1e-8))}, "
          f"#zero={int(np.sum(np.abs(wb)<1e-8))}")

    print("\n--- (C) SOFTENING scan: close the locking couplings, watch the gap ---")
    print("  (3x3, alpha=2, nu=1 fixed; vary modulus-lock delta and degree-lock gamma)")
    print(f"  {'delta':>7} {'gamma':>7} {'lowest eig':>12} {'2nd lowest':>12} {'#neg':>6}")
    scan = []
    for (dd, gg) in [(10, 10), (1, 10), (0.1, 10), (0.01, 10), (0, 10),
                     (10, 1), (10, 0.1), (10, 0.01), (10, 0),
                     (1, 1), (0.1, 0.1), (0, 0)]:
        fun = lambda x, a=2.0, g=gg, de=dd: full(x, a, g, de, 1.0)
        Hs = finite_diff_hessian(fun, D_to_x(D3))
        ws = np.linalg.eigvalsh(Hs)
        nneg = int(np.sum(ws < -1e-8))
        scan.append({"delta": dd, "gamma": gg, "lowest": float(ws[0]),
                     "second": float(ws[1]), "n_neg": nneg})
        print(f"  {dd:>7} {gg:>7} {ws[0]:>12.4e} {ws[1]:>12.4e} {nneg:>6}")

    out = ROOT / "experiments" / "exp_gravity_mass_gap_last_run.json"
    out.write_text(json.dumps({"D0_spectrum": [float(lam[0]), float(lam[-1])],
                               "simple_scan": rows,
                               "crosscheck_maxdiff": float(np.max(np.abs(w3a - w3f))),
                               "full_3x3_lowest": float(wb[0]),
                               "full_3x3_nneg": int(np.sum(wb < -1e-8)),
                               "softening_scan": scan}, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
