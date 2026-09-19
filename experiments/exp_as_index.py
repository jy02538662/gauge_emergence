"""检验：离散指标定理（Atiyah–Singer 格点版）——绕数 = 零模数，不依赖连续极限。

坐实「拓扑是离散不变量」（用户的关键洞察：拓扑这一半不需要破墙）。
1D 手征 Dirac：Q(k) = e^{ik} - 1 + m。
  - 拓扑指标（绕数）= arg Q(k) 绕原点圈数（k 扫过布里渊区）。
  - 分析指标（零模数）= 开放边界 Q 矩阵的零模（最小奇异值 ≈ 0）。
  - 指标定理：绕数 = 零模数（离散格点上直接成立，不取连续极限）。

Code: `py -m experiments.exp_as_index`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def winding_number(m, nk=2000):
    """绕数（拓扑指标）= arg Q(k) 绕原点圈数，Q(k)=e^{ik}-1+m。"""
    k = np.linspace(0, 2 * np.pi, nk, endpoint=False)
    Q = np.exp(1j * k) - 1 + m
    arg = np.unwrap(np.angle(Q))
    return round((arg[-1] - arg[0]) / (2 * np.pi))


def zero_mode_present(m, N=60):
    """零模是否存在（分析指标）：Q 矩阵（上双对角：对角 m-1，上对角 1）最小奇异值 ≈ 0？"""
    Q = np.zeros((N, N))
    for i in range(N):
        Q[i, i] = m - 1
        if i + 1 < N:
            Q[i, i + 1] = 1.0
    s = np.linalg.svd(Q, compute_uv=False)
    return float(s[-1])  # 最小奇异值


def main():
    print("=== 离散指标定理：绕数 = 零模数（不依赖连续极限）===")
    print()

    # ---- 扫 m，对比绕数（拓扑）和零模（分析）----
    print("1. 绕数（拓扑指标）vs 零模（分析指标），Q(k)=e^{ik}-1+m（相变点 m=0,2 除外）")
    print(f"   {'m':<6} {'绕数':<6} {'最小奇异值':<14} {'零模':<6} {'绕数=零模'}")
    results = {}
    for m in [-1.0, 0.5, 1.0, 1.5, 3.0]:
        w = winding_number(m)
        smin = zero_mode_present(m)
        has_zero = smin < 1e-6
        match = (w == (1 if has_zero else 0))
        results[str(m)] = {"winding": w, "min_singular": smin, "zero_mode": has_zero}
        print(f"   {m:<6} {w:<6} {smin:<14.2e} {('是' if has_zero else '否'):<6} {('[OK]' if match else '[FAIL]')}")

    # ---- 结论 ----
    print()
    print("=== 结论 ===")
    print("  绕数（拓扑）= 零模数（分析），在离散格点上直接成立，不取连续极限。")
    print("  => 拓扑荷（绕数）是「纯离散不变量」，不需要破「离散→连续」的墙。")
    print("  => 墙的两半不对称：拓扑（离散完整）vs 光滑（才需连续极限）。")

    summary = {
        "question": "does winding number = zero modes on the lattice (discrete index theorem)?",
        "results": results,
        "conclusion": "winding (topological index) = zero modes (analytical index) holds directly on "
                      "the discrete lattice, WITHOUT taking the continuum limit. Topological charge "
                      "(winding) is a PURE DISCRETE invariant - it does NOT need to break the "
                      "'discrete->continuous' wall. The two halves of the wall are ASYMMETRIC: "
                      "topology (discrete, complete) vs smoothness (needs continuum limit).",
    }
    out = ROOT / "experiments" / "exp_as_index_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
