"""真算键序 T_ij = <c_i^dag c_j>（费米海非对角元，非手选），看它是否是「非对角耦合」的来源。

1D 环 + 缺陷（局域势 V），费米海密度矩阵 rho（填负能）。
键序 T_i = rho_{i,i+1}（近邻非对角元）。
均匀环 T_i 平移不变；缺陷环 T_i 位置依赖（缺陷处变化）。

这是「非对角耦合」（Weyl 的原料）的真算来源，不是手选 T(q)。

Code: `py -m experiments.exp_verify_bond_order_real`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def ring_D(N, V=0.0, x0=None):
    D = np.zeros((N, N))
    for i in range(N):
        j = (i + 1) % N
        D[i, j] = 1.0; D[j, i] = 1.0
    if x0 is not None:
        D[x0, x0] += V
    return D


def bond_order(D):
    ev, Vm = np.linalg.eigh(D)
    occ = ev < 0.0
    P = Vm[:, occ] @ Vm[:, occ].conj().T
    N = D.shape[0]
    T = np.array([np.real(P[i, (i + 1) % N]) for i in range(N)])
    return T


def main():
    print("=== 真算键序 T_ij = <c_i^dag c_j>（非对角元，非手选）===")
    print()

    N = 120
    x0 = 60
    V = 0.8

    D0 = ring_D(N)                 # 均匀环
    D1 = ring_D(N, V=V, x0=x0)     # 缺陷环

    T0 = bond_order(D0)
    T1 = bond_order(D1)

    print(f"  均匀环：键序 T_i（近邻非对角元）mean={T0.mean():.5f}, std={T0.std():.2e}")
    print(f"    => T_i 平移不变（均匀环）。")
    print()
    print(f"  缺陷环（V={V} at x0={x0}）：键序 T_i mean={T1.mean():.5f}, std={T1.std():.4f}")
    print(f"    缺陷附近 T_i：{[f'{T1[(x0+d)%N]:.4f}' for d in (-3,-2,-1,0,1,2,3)]}")
    print(f"    => T_i 在缺陷处位置依赖（非对角耦合，真算来源）。")

    print()
    print("=== 结论 ===")
    print("  键序 T_i（非对角元）是真算的（Hellmann-Feynman），位置依赖来自缺陷（物质源）。")
    print("  这是「非对角耦合」（Weyl 的原料），不手选 T(q)、不手选 lambda。")
    print("  但 Weyl 需要 4D（3 空间 + 1 时间）+ 连续化（离散格点 T_i -> 连续 T_mu nu），")
    print("  这是两个墙（4 维 + 离散->连续）。")

    summary = {
        "N": N, "V": V, "x0": x0,
        "uniform_T_std": float(T0.std()),
        "defect_T_std": float(T1.std()),
        "defect_T_near": {str(d): float(T1[(x0 + d) % N]) for d in (-3, -2, -1, 0, 1, 2, 3)},
        "conclusion": "bond order T_i (off-diagonal) is real (Hellmann-Feynman), position-"
                      "dependent via defect (matter). This is the off-diagonal coupling "
                      "(Weyl ingredient), not hand-picked. But Weyl needs 4D (3+1) + "
                      "continuum (discrete T_i -> continuous T_mu nu), two walls.",
    }
    out = ROOT / "experiments" / "exp_verify_bond_order_real_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
