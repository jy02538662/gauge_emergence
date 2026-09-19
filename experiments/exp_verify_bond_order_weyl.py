"""尝试「干净缝合」第一步：键序 T_ij（非对角元）做耦合 -> Weyl != 0？

对比：对角 tau（标量缩放）-> 共形平坦（Weyl=0）；
      非对角 T_ij（键序，剪切）-> 非共形平坦（Weyl!=0，自旋 2）。

4 维度量 g = delta + eps*T，T 是对角（标量）或非对角（剪切），数值算 Weyl 范数。

Code: `py -m experiments.exp_verify_bond_order_weyl`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def weyl_norm_from_g(g0, eps=1e-4):
    """在常数度规 g0 处算 Weyl 范数（用有限差分，需 g0 附近的度规由 q 决定）。

    这里简化：直接给 4 维度量 g(q) 的闭式（g = delta + eps*T，T 是常张量），
    Weyl 由 T 的结构决定。用有限差分算。
    """
    n = 4
    # 度规是常数（T 常张量），所以 Christoffel=0，Riemann=0，Weyl=0。
    # 关键：常数度规 Weyl=0。要非零 Weyl，T 必须是 q 的函数（位置依赖）。
    # 所以这里用 T(q) = q-依赖的剪切，构造 g(q) = delta + eps * T(q)。
    return None  # 占位，下面重新设计


def main():
    print("=== 键序（非对角）做耦合 -> Weyl != 0 ===")
    print()

    # 关键洞察：常数度规 Weyl=0。要 Weyl!=0，耦合 T 必须是「位置依赖」的。
    # 键序 T_ij = <c_i^dag c_j> 在均匀系统里是平移不变的（常数），
    # 有缺陷时位置依赖。所以用「位置依赖的键序」做耦合。

    # 构造：g(q) = delta + eps * T(q)，T(q) 是位置依赖的非对角（剪切）张量。
    # 用数值有限差分算 Weyl 范数（复用之前的 weyl_norm 逻辑，这里简化）。
    def weyl_norm_g(q0, eps_t, eps_diff=1e-4):
        n = 4
        # g_ij(q) = delta_ij + eps_t * T_ij(q)
        # T 取非对角剪切 + 位置依赖：T_01 = q0*q1, T_02 = q1*q2, 等
        def G(q):
            G = np.eye(n)
            G[0, 1] = G[1, 0] = eps_t * q[0] * q[1]
            G[0, 2] = G[2, 0] = eps_t * q[1] * q[2]
            G[1, 2] = G[2, 1] = eps_t * q[2] * q[3]
            G[0, 3] = G[3, 0] = eps_t * q[3] * q[0]
            G[1, 3] = G[3, 1] = eps_t * q[0] * q[2]
            G[2, 3] = G[3, 2] = eps_t * q[1] * q[3]
            return G

        G0 = G(q0)
        Ginv = np.linalg.inv(G0)
        dG = np.zeros((n, n, n))
        for k in range(n):
            qp = q0.copy(); qp[k] += eps_diff
            qm = q0.copy(); qm[k] -= eps_diff
            dG[:, :, k] = (G(qp) - G(qm)) / (2 * eps_diff)
        Gam = np.zeros((n, n, n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    Gam[i, j, k] = 0.5 * sum(Ginv[i, l] * (dG[l, j, k] + dG[l, k, j] - dG[j, k, l]) for l in range(n))
        dGam = np.zeros((n, n, n, n))
        for l in range(n):
            qp = q0.copy(); qp[l] += eps_diff
            qm = q0.copy(); qm[l] -= eps_diff
            Gp = G(qp); Gip = np.linalg.inv(Gp)
            Gm = G(qm); Gim = np.linalg.inv(Gm)
            dGp = np.zeros((n, n, n)); dGm = np.zeros((n, n, n))
            for kk in range(n):
                qpp = qp.copy(); qpp[kk] += eps_diff; qpm = qp.copy(); qpm[kk] -= eps_diff
                dGp[:, :, kk] = (G(qpp) - G(qpm)) / (2 * eps_diff)
                qmp = qm.copy(); qmp[kk] += eps_diff; qmm = qm.copy(); qmm[kk] -= eps_diff
                dGm[:, :, kk] = (G(qmp) - G(qmm)) / (2 * eps_diff)
            Gamp = np.zeros((n, n, n)); Gamm_ = np.zeros((n, n, n))
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        Gamp[i, j, k] = 0.5 * sum(Gip[i, m] * (dGp[m, j, k] + dGp[m, k, j] - dGp[j, k, m]) for m in range(n))
                        Gamm_[i, j, k] = 0.5 * sum(Gim[i, m] * (dGm[m, j, k] + dGm[m, k, j] - dGm[j, k, m]) for m in range(n))
            dGam[:, :, :, l] = (Gamp - Gamm_) / (2 * eps_diff)
        Rm = np.zeros((n, n, n, n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        Rm[i, j, k, l] = sum(
                            G0[i, m] * (dGam[m, j, l, k] - dGam[m, j, k, l]
                                         + sum(Gam[m, p, k] * Gam[p, j, l] - Gam[m, p, l] * Gam[p, j, k] for p in range(n)))
                            for m in range(n))
        Ric = np.zeros((n, n))
        for j in range(n):
            for l in range(n):
                Ric[j, l] = sum(Rm[i, j, i, l] for i in range(n))
        R = sum(Ginv[j, l] * Ric[j, l] for j in range(n) for l in range(n))
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

    q0 = np.array([0.5, 0.5, 0.5, 0.5])
    # eps_t = 0（对角 delta）vs eps_t = 0.3（非对角剪切）
    w0 = weyl_norm_g(q0, 0.0)
    w1 = weyl_norm_g(q0, 0.3)
    print(f"  eps=0（对角 δ，标量）：Weyl 范数 = {w0:.2e}")
    print(f"  eps=0.3（非对角键序，剪切）：Weyl 范数 = {w1:.2e}")
    print(f"  => 非对角键序打开 Weyl：{w1 > 100 * max(w0, 1e-12)}")

    print()
    print("=== 诚实边界 ===")
    print("  非对角键序（剪切）-> Weyl != 0，区别于对角 tau（标量）-> Weyl=0。")
    print("  但这里 T(q) 仍是「手选的位置依赖剪切」，不是从密度矩阵非对角元真算。")
    print("  下一步：T_ij = <c_i^dag c_j> 真算（费米海非对角元），代入度规。")

    summary = {
        "weyl_diag": float(w0),
        "weyl_offdiag": float(w1),
        "offdiag_opens_weyl": bool(w1 > 100 * max(w0, 1e-12)),
        "note": "off-diagonal bond order (shear) -> Weyl != 0, unlike diagonal tau (scalar) "
                "-> Weyl=0. But T(q) still hand-picked; next: T_ij = <c_i^dag c_j> real.",
    }
    out = ROOT / "experiments" / "exp_verify_bond_order_weyl_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
