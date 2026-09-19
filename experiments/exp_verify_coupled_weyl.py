"""数值验证「耦合 lambda -> Weyl != 0」（自旋 2 来源）。

4 参数指数族 A = sum q^2 + lambda*q1*q2*q3*q4，Fisher 度规 G = Hessian(A)。
数值（有限差分）算 Weyl 张量的 Frobenius 范数：
lambda=0 -> Weyl=0（共形平坦）；lambda!=0 -> Weyl!=0（自旋 2）。

Code: `py -m experiments.exp_verify_coupled_weyl`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def G_matrix(q, lam):
    q1, q2, q3, q4 = q
    G = np.zeros((4, 4))
    G[0, 0] = G[1, 1] = G[2, 2] = G[3, 3] = 2.0
    G[0, 1] = G[1, 0] = lam * q3 * q4
    G[0, 2] = G[2, 0] = lam * q2 * q4
    G[0, 3] = G[3, 0] = lam * q2 * q3
    G[1, 2] = G[2, 1] = lam * q1 * q4
    G[1, 3] = G[3, 1] = lam * q1 * q3
    G[2, 3] = G[3, 2] = lam * q1 * q2
    return G


def weyl_norm(q0, lam, eps=1e-4):
    """数值算 Weyl 张量（全协变）的 Frobenius 范数，在 q0 处。"""
    n = 4
    G0 = G_matrix(q0, lam)
    Ginv = np.linalg.inv(G0)

    # 一阶导 dG[i,j,k] = dG_ij/dq_k
    dG = np.zeros((n, n, n))
    for k in range(n):
        qp = q0.copy(); qp[k] += eps
        qm = q0.copy(); qm[k] -= eps
        dG[:, :, k] = (G_matrix(qp, lam) - G_matrix(qm, lam)) / (2 * eps)

    # Christoffel Gamma[i,j,k]（第一类，降指标）和 Gamma^i_jk（第二类）
    # 用全协变 Riemann，需要第二类 Christoffel
    Gam = np.zeros((n, n, n))  # Gamma^i_jk
    for i in range(n):
        for j in range(n):
            for k in range(n):
                Gam[i, j, k] = 0.5 * sum(
                    Ginv[i, l] * (dG[l, j, k] + dG[l, k, j] - dG[j, k, l]) for l in range(n))

    # 二阶导 ddG[i,j,k,l] = d^2G_ij/dq_k dq_l
    ddG = np.zeros((n, n, n, n))
    for k in range(n):
        for l in range(n):
            qpp = q0.copy(); qpp[k] += eps; qpp[l] += eps
            qpm = q0.copy(); qpm[k] += eps; qpm[l] -= eps
            qmp = q0.copy(); qmp[k] -= eps; qmp[l] += eps
            qmm = q0.copy(); qmm[k] -= eps; qmm[l] -= eps
            ddG[:, :, k, l] = (G_matrix(qpp, lam) - G_matrix(qpm, lam)
                               - G_matrix(qmp, lam) + G_matrix(qmm, lam)) / (4 * eps ** 2)

    # Riemann 全协变 R[i,j,k,l] = g_im (d_k Gam^m_jl - d_l Gam^m_jk + Gam^m_pk Gam^p_jl - Gam^m_pl Gam^p_jk)
    # 需要 dGam（Christoffel 的导数）
    dGam = np.zeros((n, n, n, n))  # dGam[i,j,k,l] = d Gam^i_jk / dq_l
    for l in range(n):
        qp = q0.copy(); qp[l] += eps
        qm = q0.copy(); qm[l] -= eps
        # 重新算 Christoffel 在 qp, qm
        Gp = G_matrix(qp, lam); Gip = np.linalg.inv(Gp)
        Gm = G_matrix(qm, lam); Gim = np.linalg.inv(Gm)
        dGp = np.zeros((n, n, n)); dGm = np.zeros((n, n, n))
        for kk in range(n):
            qpp = qp.copy(); qpp[kk] += eps
            qpm = qp.copy(); qpm[kk] -= eps
            dGp[:, :, kk] = (G_matrix(qpp, lam) - G_matrix(qpm, lam)) / (2 * eps)
            qmp = qm.copy(); qmp[kk] += eps
            qmm = qm.copy(); qmm[kk] -= eps
            dGm[:, :, kk] = (G_matrix(qmp, lam) - G_matrix(qmm, lam)) / (2 * eps)
        Gamp = np.zeros((n, n, n)); Gamm_ = np.zeros((n, n, n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    Gamp[i, j, k] = 0.5 * sum(Gip[i, m] * (dGp[m, j, k] + dGp[m, k, j] - dGp[j, k, m]) for m in range(n))
                    Gamm_[i, j, k] = 0.5 * sum(Gim[i, m] * (dGm[m, j, k] + dGm[m, k, j] - dGm[j, k, m]) for m in range(n))
        dGam[:, :, :, l] = (Gamp - Gamm_) / (2 * eps)

    # Riemann 全协变
    Rm = np.zeros((n, n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    Rm[i, j, k, l] = sum(
                        G0[i, m] * (dGam[m, j, l, k] - dGam[m, j, k, l]
                                     + sum(Gam[m, p, k] * Gam[p, j, l] - Gam[m, p, l] * Gam[p, j, k] for p in range(n)))
                        for m in range(n))

    # Ricci
    Ric = np.zeros((n, n))
    for j in range(n):
        for l in range(n):
            Ric[j, l] = sum(Rm[i, j, i, l] for i in range(n))
    # 标量曲率
    R = sum(Ginv[j, l] * Ric[j, l] for j in range(n) for l in range(n))

    # Weyl 全协变
    W = np.zeros((n, n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    W[i, j, k, l] = (Rm[i, j, k, l]
                                     - (G0[i, k] * Ric[j, l] - G0[i, l] * Ric[j, k]
                                        - G0[j, k] * Ric[i, l] + G0[j, l] * Ric[i, k]) / 2
                                     + R * (G0[i, k] * G0[j, l] - G0[i, l] * G0[j, k]) / 6)
    return np.sqrt((W ** 2).sum())


def main():
    print("=== 数值验证「耦合 lambda -> Weyl != 0」===")
    print()

    q0 = np.array([0.5, 0.5, 0.5, 0.5])

    norm0 = weyl_norm(q0, 0.0)
    norm1 = weyl_norm(q0, 1.0)
    print(f"  lambda=0（无耦合）：Weyl 范数 = {norm0:.2e}")
    print(f"  lambda=1（耦合）  ：Weyl 范数 = {norm1:.2e}")
    print(f"  => 耦合打开 Weyl：{norm1 > 100 * max(norm0, 1e-12)}（norm1/norm0 = {norm1/max(norm0,1e-12):.1e}）")

    print()
    print("=== 结论 ===")
    print("  lambda=0（无耦合）-> Weyl=0（共形平坦，标量）。")
    print("  lambda!=0（耦合）-> Weyl!=0（自旋 2 涌现）。")
    print("  => 「观察者之间的耦合 lambda」是「自旋 2 弯曲（Weyl!=0）」的来源。")

    summary = {
        "weyl_norm_lambda0": float(norm0),
        "weyl_norm_lambda1": float(norm1),
        "coupling_opens_weyl": bool(norm1 > 100 * max(norm0, 1e-12)),
        "conclusion": "lambda (observer coupling) opens Weyl != 0 (spin-2). "
                      "Next: derive lambda from theory (observer relation), not hand-picked.",
    }
    out = ROOT / "experiments" / "exp_verify_coupled_weyl_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
