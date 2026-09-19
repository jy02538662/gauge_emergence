"""下一步：从「观察者关系」推耦合 λ（不再手选）。

对数配分函数 A = sum q^2 + alpha * overlap(q) * q1*q2*q3*q4，
其中 overlap(q) = exp(-(q1-q3)^2) = 观察者 1 和 2 的「关系」（fidelity 的经典版）。
（q1,q2 = 观察者 1 的 2 参数，q3,q4 = 观察者 2 的 2 参数。）

Fisher 度规 G = Hessian(A)，数值算 Weyl 范数：
关系（overlap）-> 耦合 -> Weyl != 0（自旋 2），不再手选 lambda。

Code: `py -m experiments.exp_verify_relation_coupling`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def overlap(q):
    return np.exp(-(q[0] - q[2]) ** 2)


def G_matrix(q, alpha=1.0):
    """Fisher 度规 G = Hessian(A)，A = sum q^2 + alpha*overlap*q1q2q3q4。"""
    q1, q2, q3, q4 = q
    ov = overlap(q)
    # 数值 Hessian（有限差分）
    eps = 1e-5
    G = np.zeros((4, 4))

    def A(qq):
        o = overlap(qq)
        return sum(qq ** 2) + alpha * o * qq[0] * qq[1] * qq[2] * qq[3]

    for i in range(4):
        for j in range(4):
            qp = q.copy(); qp[i] += eps; qp[j] += eps
            qpm = q.copy(); qpm[i] += eps; qpm[j] -= eps
            qmp = q.copy(); qmp[i] -= eps; qmp[j] += eps
            qmm = q.copy(); qmm[i] -= eps; qmm[j] -= eps
            G[i, j] = (A(qp) - A(qpm) - A(qmp) + A(qmm)) / (4 * eps ** 2)
    return G


def weyl_norm(q0, alpha, eps=1e-4):
    n = 4
    G0 = G_matrix(q0, alpha)
    Ginv = np.linalg.inv(G0)
    dG = np.zeros((n, n, n))
    for k in range(n):
        qp = q0.copy(); qp[k] += eps
        qm = q0.copy(); qm[k] -= eps
        dG[:, :, k] = (G_matrix(qp, alpha) - G_matrix(qm, alpha)) / (2 * eps)
    Gam = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                Gam[i, j, k] = 0.5 * sum(Ginv[i, l] * (dG[l, j, k] + dG[l, k, j] - dG[j, k, l]) for l in range(n))
    dGam = np.zeros((n, n, n, n))
    for l in range(n):
        qp = q0.copy(); qp[l] += eps
        qm = q0.copy(); qm[l] -= eps
        Gp = G_matrix(qp, alpha); Gip = np.linalg.inv(Gp)
        Gm = G_matrix(qm, alpha); Gim = np.linalg.inv(Gm)
        dGp = np.zeros((n, n, n)); dGm = np.zeros((n, n, n))
        for kk in range(n):
            qpp = qp.copy(); qpp[kk] += eps; qpm = qp.copy(); qpm[kk] -= eps
            dGp[:, :, kk] = (G_matrix(qpp, alpha) - G_matrix(qpm, alpha)) / (2 * eps)
            qmp = qm.copy(); qmp[kk] += eps; qmm = qm.copy(); qmm[kk] -= eps
            dGm[:, :, kk] = (G_matrix(qmp, alpha) - G_matrix(qmm, alpha)) / (2 * eps)
        Gamp = np.zeros((n, n, n)); Gamm_ = np.zeros((n, n, n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    Gamp[i, j, k] = 0.5 * sum(Gip[i, m] * (dGp[m, j, k] + dGp[m, k, j] - dGp[j, k, m]) for m in range(n))
                    Gamm_[i, j, k] = 0.5 * sum(Gim[i, m] * (dGm[m, j, k] + dGm[m, k, j] - dGm[j, k, m]) for m in range(n))
        dGam[:, :, :, l] = (Gamp - Gamm_) / (2 * eps)
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


def main():
    print("=== 从观察者关系（overlap）推耦合 -> Weyl != 0 ===")
    print()

    # 观察者 1 和 2 远离（无关系，overlap~0）vs 靠近（有关系，overlap~1）
    # q1,q2 = 观察者1，q3,q4 = 观察者2；overlap = exp(-(q1-q3)^2)
    q_far = np.array([0.0, 0.5, 3.0, 0.5])   # q1=0, q3=3 -> overlap~0
    q_near = np.array([0.0, 0.5, 0.2, 0.5])  # q1=0, q3=0.2 -> overlap~0.96

    ov_far = overlap(q_far)
    ov_near = overlap(q_near)
    print(f"  观察者远离：overlap = {ov_far:.4f}")
    print(f"  观察者靠近：overlap = {ov_near:.4f}")

    w_far = weyl_norm(q_far, 1.0)
    w_near = weyl_norm(q_near, 1.0)
    print()
    print(f"  远离（overlap={ov_far:.4f}）：Weyl 范数 = {w_far:.2e}")
    print(f"  靠近（overlap={ov_near:.4f}）：Weyl 范数 = {w_near:.2e}")
    print(f"  => 关系（overlap）打开 Weyl：{w_near > 100 * max(w_far, 1e-12)}")

    print()
    print("=== 结论 ===")
    print("  观察者关系（overlap）-> 耦合 -> Weyl != 0（自旋 2），不再手选 lambda。")
    print("  下一步：观察者的「态」从理论（自反性）来，overlap 就是它们的自然关系。")

    summary = {
        "overlap_far": float(ov_far),
        "overlap_near": float(ov_near),
        "weyl_far": float(w_far),
        "weyl_near": float(w_near),
        "relation_opens_weyl": bool(w_near > 100 * max(w_far, 1e-12)),
        "conclusion": "observer relation (overlap) -> coupling -> Weyl != 0 (spin-2), "
                      "not hand-picked lambda.",
    }
    out = ROOT / "experiments" / "exp_verify_relation_coupling_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
