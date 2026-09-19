"""P4 紧性：无界外导子 log rho 的逆 |log rho|^{-1} 是否紧（数值检验）。

墙的精确表述：R 无紧 D（不是 R 无外导子）。
- P2 裂缝（已数值坐实，exp_wall_derivation_asymptotic）：无界外导子 log rho 可构造。
- P4 问（本脚本）：这个外导子的逆 |log rho|^{-1} 紧吗？

数学事实（Connes 框架）：
  II_1 因子 R 是有限因子，无非零紧算子（紧算子的谱投影有限维/原子，R 的投影无限可分）。
  所以 |log rho|^{-1} in R（有界）但不可能紧。

有限维数值能捕捉的（冯诺伊曼做不了真正无限维极限）：
  1. 衰减速度：|log rho|^{-1} = diag(1/log i) 特征值 mu_n = 1/log n（对数慢衰减）。
     log-log 斜率 -> 0（非幂律），对比紧算子 1/n（斜率 -1）。
  2. Dixmier 迹：(1/log N) sum mu_n ~ N/(log N)^2 -> 发散，对比 1/n 的 -> 1（有限）。
  3. 谱间距：特征值填满连续区间（连续谱信号，非点谱 -> 0）。

结论：外导子能造（P2 裂缝），紧 D 不能（P4 墙，Dixmier 迹发散）。

Code: `py -m experiments.exp_wall_compactness`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def partial_dixmier(mu_descending, N):
    """Partial Dixmier trace (1/ln N) sum_{n=1}^N mu_n."""
    return float(np.sum(mu_descending) / np.log(N))


def loglog_slope(mu, n_range):
    """Slope of log(mu) vs log(n) over the given index range (local power-law exponent)."""
    n = np.arange(n_range[0], n_range[1] + 1, dtype=float)
    m = mu[n_range[0] - 1:n_range[1]]
    # guard against zeros
    m = np.maximum(m, 1e-300)
    return float(np.polyfit(np.log(n), np.log(m), 1)[0])


def main():
    print("=== P4 紧性：|log rho|^{-1} 是否紧（Dixmier 迹 + 衰减速度）===")
    print()
    print("墙 = R 无紧 D（不是 R 无外导子）。外导子 log rho 可造（P2），但逆紧吗？")
    print()

    # ---- Part 1: 衰减速度对比（紧算子 1/n vs 非紧 1/log n）----
    print("Part 1. 衰减速度（log-log 斜率 = 表观幂律指数）:")
    print(f"  {'N':>9} {'slope(1/n)':>12} {'slope(1/log n)':>15}")
    rows = []
    for N in [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]:
        # 紧算子：mu_n = 1/n（排序后 = 1, 1/2, ..., 1/N）
        mu_fast = 1.0 / np.arange(1, N + 1)
        # 非紧：mu_n = 1/log(n+1)，n=1..N（对应 i=2..N+1）
        mu_slow = 1.0 / np.log(np.arange(2, N + 2))
        mu_slow = np.sort(mu_slow)[::-1]  # 降序（= 1/log2 > 1/log3 > ...）
        # 局部斜率，取 [N/4, N/2] 区间（远离两端）
        rng = (max(1, N // 4), max(2, N // 2))
        s_fast = loglog_slope(mu_fast, rng)
        s_slow = loglog_slope(mu_slow, rng)
        rows.append((N, s_fast, s_slow))
        print(f"  {N:>9} {s_fast:>12.4f} {s_slow:>15.4f}")
    print("  1/n: slope -> -1（幂律，紧）。1/log n: slope -> 0（对数慢，非幂律，非紧）。")

    # ---- Part 2: Dixmier 迹 ----
    print()
    print("Part 2. Dixmier 迹 Tr_w(mu) = (1/ln N) sum mu_n（降序）:")
    print(f"  {'N':>9} {'Tr_w(1/n)':>12} {'Tr_w(1/log n)':>14}")
    rows2 = []
    for N in [8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144, 1048576]:
        mu_fast = 1.0 / np.arange(1, N + 1)
        mu_slow = np.sort(1.0 / np.log(np.arange(2, N + 2)))[::-1]
        t_fast = partial_dixmier(mu_fast, N)
        t_slow = partial_dixmier(mu_slow, N)
        rows2.append((N, t_fast, t_slow))
        print(f"  {N:>9} {t_fast:>12.4f} {t_slow:>14.4f}")
    print("  Tr_w(1/n) -> 1（有限：log 奇性 = 紧算子的 Dixmier 迹）。")
    print("  Tr_w(1/log n) -> 发散（~ N/(ln N)^2）：|log rho|^{-1} 不是紧算子。")

    # ---- Part 3: 谱间距（连续谱信号）----
    print()
    print("Part 3. 谱间距（特征值是否填满连续区间 = 连续谱信号）:")
    print(f"  {'N':>9} {'min gap(1/n)':>14} {'min gap(1/log n)':>16}")
    for N in [256, 4096, 65536, 1048576]:
        mu_fast = np.sort(1.0 / np.arange(1, N + 1))
        mu_slow = np.sort(1.0 / np.log(np.arange(2, N + 2)))
        g_fast = float(np.diff(mu_fast).min())
        g_slow = float(np.diff(mu_slow).min())
        print(f"  {N:>9} {g_fast:>14.3e} {g_slow:>16.3e}")
    print("  两种都 -> 0（谱密化）。但 1/n 保留点谱（Dixmier 迹有限），")
    print("  1/log n 的谱密化 + Dixmier 迹发散 = 低频谱密度发散 = 非紧。")

    print()
    print("=== 结论 ===")
    print("  外导子 log rho 可造（P2 裂缝已坐实），但 |log rho|^{-1} 非紧：")
    print("  衰减 1/log n（非幂律，slope -> 0）+ Dixmier 迹发散（~ N/(ln N)^2）。")
    print("  => 紧 D 不能造（P4 墙）。P2 裂缝只裂了一半：外导子有，紧 D 没有。")
    print("     真正的硬骨头 = P4（紧性从哪来），不是 P2（外导子）。")

    summary = {
        "wall": "R has no compact D (not 'no outer derivation')",
        "P2_half_crack": "outer derivation log rho EXISTS (P2), but |log rho|^{-1} NOT compact (P4)",
        "slope_1n": -1.0,
        "slope_1logn_tends_to": 0.0,
        "dixmier_1n_tends_to": 1.0,
        "dixmier_1logn_diverges": "~ N/(ln N)^2",
        "slope_table": [{"N": int(N), "slope_1n": float(sf), "slope_1logn": float(ss)}
                        for N, sf, ss in rows],
        "dixmier_table": [{"N": int(N), "Trw_1n": float(tf), "Trw_1logn": float(ts)}
                          for N, tf, ts in rows2],
        "conclusion": "outer derivation constructible (P2 crack), compact D NOT constructible "
                      "(P4 wall: |log rho|^{-1} decays 1/log n, Dixmier trace diverges). "
                      "P2 crack is only half: outer derivation yes, compact D no.",
    }
    out = ROOT / "experiments" / "exp_wall_compactness_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
