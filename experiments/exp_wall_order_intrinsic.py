"""P3 换墙：差分 D 需要的一维空间序，是内生还是外给（负结果演示）。

P3 换路（exp_wall_connes_distance_euclidean 已坐实）：谱距离从差分 D 恢复欧氏距离。
但差分 D 要求一维序（i 和 i+1 空间相邻）。这个序从哪来？

关键区分：
  - 时间序：模流 sigma_t = Ad(rho^{it}) 的 t 单调（内生，自反性的内禀结构）。
  - 空间序：差分 D 的 i 相邻（外给，标号手放）。

检验：log rho = diag(-log i) 的单调性来自标号 i 的顺序；随机排列标号后破坏。
=> 自反性（厄米性 + 离散谱）不锁空间序。
=> P3 换路 = 从「Aut->Diff」换成「空间序从哪来」，没破墙，是换墙。

Code: `py -m experiments.exp_wall_order_intrinsic`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sign_flips(x):
    """相邻元素差分的符号翻转次数（0 = 全单调，越大越无序）。"""
    d = np.diff(x)
    d = d[np.abs(d) > 1e-12]
    if d.size < 2:
        return 0
    return int(np.sum(np.sign(d[1:]) != np.sign(d[:-1])))


def main():
    print("=== P3 换墙：差分 D 的空间序，内生还是外给 ===")
    print()

    N = 32
    i = np.arange(1, N + 1, dtype=float)
    logrho = -np.log(i)                    # 对角元 -log i（标号序，单调递减）
    flips_ordered = sign_flips(logrho)
    print(f"1. log rho = diag(-log i)（标号 i 的顺序）：")
    print(f"   对角元单调递减，差分符号翻转次数 = {flips_ordered}（0 = 全单调 = 有序）")

    # 随机排列标号
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    logrho_perm = -np.log(i[perm])         # 排列后的对角元 -log pi(i)
    flips_perm = sign_flips(logrho_perm)
    print()
    print(f"2. 随机排列标号后，对角元 = -log pi(i)：")
    print(f"   差分符号翻转次数 = {flips_perm}（>0 = 无序）")
    print(f"   => 单调性（序）来自标号 i 的手放，排列即破坏。")

    # 自反性只给厄米性，不锁序：对同一谱，标号可以任意重排
    print()
    print("3. 自反性 D_ij = D_ji* 只给厄米性 + 离散谱，不锁一维序：")
    print("   同一个谱 {log i}，标号可任意重排（permutation 是 D 的自同构），")
    print("   谱不变、序全变 => 空间序不是自反性内生的。")

    print()
    print("=== 结论 ===")
    print("  时间序（模流 sigma_t 的 t 单调）= 内生。")
    print("  空间序（差分 D 的 i 相邻）= 外给（标号手放）。")
    print("  => P3 换路 = 从「Aut->Diff」换成「空间序从哪来」，没破墙，是换墙。")

    summary = {
        "ordered_flips": flips_ordered,
        "permuted_flips": flips_perm,
        "time_order": "intrinsic (modular flow t monotone)",
        "space_order": "extrinsic (label i hand-picked)",
        "conclusion": "Difference D needs a 1D SPATIAL order (i adjacent to i+1), which is "
                      "EXTRINSIC (label hand-picked, permutation breaks it). Self-reflexivity "
                      "(Hermiticity + discrete spectrum) does NOT fix spatial order. So P3 "
                      "'reroute' replaces 'Aut->Diff' with 'where does spatial order come from' "
                      "-- not a wall-break, a wall-swap.",
    }
    out = ROOT / "experiments" / "exp_wall_order_intrinsic_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
