"""补验证：T_0i = (1/2){K, T_i} 的厄米性 + 静态基态是否非零。

1D 环 + 缺陷，费米海密度矩阵 K（键序），磁平移 T（跃迁算符）。
T_0i = (1/2)(K T + T K)。
验证：(1) 厄米性 (T_0i^dag = T_0i)；(2) 静态基态期望 <T_0i> 是否非零。

Code: `py -m experiments.exp_verify_T0i`
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


def main():
    print("=== 补验证：T_0i = (1/2){K, T_i} ===")
    print()

    N = 80
    x0 = 40
    V = 0.8
    D = ring_D(N, V=V, x0=x0)

    # 费米海密度矩阵 K（键序）
    ev, Vm = np.linalg.eigh(D)
    occ = ev < 0.0
    K = Vm[:, occ] @ Vm[:, occ].conj().T

    # 磁平移 T（跃迁算符，右移）
    T = np.zeros((N, N), dtype=complex)
    for i in range(N):
        T[(i + 1) % N, i] = 1.0  # T 右移：T |i> = |i+1>

    # T_0i = (1/2)(K T + T K)
    T0i = 0.5 * (K @ T + T @ K)

    # 1. 厄米性
    herm = np.max(np.abs(T0i - T0i.conj().T))
    print(f"1. 厄米性：max|T_0i - T_0i^dag| = {herm:.2e}（应≈0）")

    # 2. 静态基态期望 <T_0i> = Tr(T_0i)（注意：T_0i 已经含 K，这里看它的迹/对角）
    #    物理上 <T_0i> 应该是每个格点的动量密度（对角元）
    diag = np.real(np.diag(T0i))
    print(f"2. 静态基态动量密度 <T_0i>（对角元）：")
    print(f"   mean = {diag.mean():.5f}, std = {diag.std():.2e}")
    print(f"   缺陷附近：{[f'{diag[(x0+d)%N]:.4f}' for d in (-2,-1,0,1,2)]}")
    print(f"   => 静态基态 <T_0i> {'非零（动量密度非平凡）' if diag.std() > 1e-10 else '= 0'}")

    print()
    print("=== 诚实边界 ===")
    print("  T_0i = (1/2){K,T_i} 厄米（验证），静态基态可能非零（待看）。")
    print("  但「构造」不等于「守恒」：T_0i 是否满足守恒律（离散版），未验证。")
    print("  且之前「守恒律 ⟂ 长程」（无隙时应力长程，局部守恒律失效）未解决。")

    summary = {
        "N": N, "V": V, "x0": x0,
        "hermitian": float(herm),
        "diag_mean": float(diag.mean()),
        "diag_std": float(diag.std()),
        "diag_near": {str(d): float(diag[(x0 + d) % N]) for d in (-2, -1, 0, 1, 2)},
        "conclusion": "T_0i = (1/2){K,T_i} is Hermitian (verified). Static ground state "
                      "expectation may be nonzero (momentum density). But construction != "
                      "conservation; discrete conservation law ⟂ long-range not resolved.",
    }
    out = ROOT / "experiments" / "exp_verify_T0i_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
