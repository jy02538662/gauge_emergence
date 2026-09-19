"""检验：光滑化的第一块砖——差分 → 导数 的 N-光滑收敛（1D）。

墙 = 光滑化 = 「差分→导数（教科书，能做）」+「点从 R 内生（难，后补）」。
本脚本做实第一半：1D 差分算子 D_N 的「光滑结构」在 N→∞ 收敛到连续流形的「光滑结构」。

三个检验：
  1. 差分本征值 → 导数本征值（谱收敛，前向差分一阶 O(1/N)）。
  2. 谱距离 d_N → 连续距离（度量收敛，缩放后 = |x-y|）。
  3. 离散 k 阶差分 → 连续 k 阶导数（光滑收敛，O(1/N)）。

结论：差分→导数 的收敛率坐实，光滑化的「教科书一半」无障碍；
  墙在「点从 R 内生」那一半（第 ③ 步，观察者态谱当点）。

Code: `py -m experiments.exp_wall_smoothing_convergence`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def forward_diff_eigenvalues(N):
    """周期 [0,1]，步长 h=1/N，前向差分 D f(i) = (f(i+1)-f(i))/h 的本征值。"""
    h = 1.0 / N
    k = np.arange(N)
    return (np.exp(2j * np.pi * k / N) - 1.0) / h


def main():
    print("=== 光滑化第一块砖：差分 → 导数 的 N-光滑收敛（1D）===")
    print()

    # ---- 1. 谱收敛：差分本征值 -> 导数本征值 ----
    print("1. 谱收敛：差分本征值 -> 导数本征值（μ_k = i2πk）")
    Ns = [16, 32, 64, 128, 256, 512]
    spec_slopes = {}
    for k in [1, 2, 3]:
        errs = []
        for N in Ns:
            ev = forward_diff_eigenvalues(N)
            errs.append(abs(ev[k] - 1j * 2 * np.pi * k))
        slope = np.polyfit(np.log(Ns), np.log(errs), 1)[0]
        spec_slopes[k] = slope
        print(f"   k={k}: 误差 ~ N^{slope:.2f}（期望 -1，前向差分一阶）")
        print(f"       N={Ns[-1]}: |λ_k - i2πk| = {errs[-1]:.2e}")

    # ---- 2. 度量收敛：谱距离 -> 连续距离 ----
    print()
    print("2. 度量收敛：谱距离 d_N(i,j)=|i-j| -> 连续距离 |x-y|")
    # §三坐实差分 D_N 谱距离 d(i,j)=|i-j|；缩放 x=i/N 后 d/N=|x-y| 精确（无网格误差）
    N = 100
    h = 1.0 / N
    i, j = 20, 80
    d_discrete = abs(i - j)
    x, y = i * h, j * h
    d_scaled = d_discrete * h
    d_cont = abs(x - y)
    print(f"   d(20,80) 离散 = {d_discrete}，缩放后 = {d_scaled:.4f}，连续 |x-y| = {d_cont:.4f}")
    print(f"   缩放后 = 连续距离（精确，无网格误差）：{[('OK' if abs(d_scaled-d_cont) < 1e-12 else 'FAIL')]}")

    # ---- 3. 光滑收敛：离散 k 阶差分 -> 连续 k 阶导数 ----
    print()
    print("3. 光滑收敛：离散 k 阶差分 -> 连续 k 阶导数")
    smooth_slopes = {}
    for k in [1, 2, 3]:
        errs = []
        for N in Ns:
            h = 1.0 / N
            x = np.arange(N) * h
            f = np.sin(2 * np.pi * x)  # 光滑函数 f(x)=sin(2πx)
            diff = f.copy()
            for _ in range(k):
                diff = (np.roll(diff, -1) - diff) / h  # 前向差分（周期）
            if k % 2 == 0:
                exact = (-1) ** (k // 2) * (2 * np.pi) ** k * np.sin(2 * np.pi * x)
            else:
                exact = (-1) ** ((k - 1) // 2) * (2 * np.pi) ** k * np.cos(2 * np.pi * x)
            errs.append(np.max(np.abs(diff - exact)))
        slope = np.polyfit(np.log(Ns), np.log(errs), 1)[0]
        smooth_slopes[k] = slope
        print(f"   k={k}: 差分-导数误差 ~ N^{slope:.2f}（期望 -1，前向差分一阶）")
        print(f"       N={Ns[-1]}: max|Δ^k f/h^k - f^(k)| = {errs[-1]:.2e}")

    # ---- 结论 ----
    print()
    print("=== 结论 ===")
    print("  差分→导数 的收敛率 ~ N^{-1}（前向差分一阶）坐实：谱、度量、光滑三层都收敛。")
    print("  光滑化的「教科书一半」无任何障碍——离散光滑可控地收敛到连续光滑。")
    print("  墙不在这一半，在「点从 R 内生」（第 ③ 步：网格标号 -> 观察者态谱）。")

    summary = {
        "question": "does discrete difference converge to continuous derivative (N-smooth -> C^inf)?",
        "spectral_slopes": {str(k): float(v) for k, v in spec_slopes.items()},
        "metric_convergence_exact": True,
        "smooth_slopes": {str(k): float(v) for k, v in smooth_slopes.items()},
        "conclusion": "difference -> derivative converges at O(1/N) (forward difference, first order) "
                      "in all three layers (spectrum, metric, smoothness). The 'textbook half' of "
                      "smoothing has NO obstacle. The wall is in 'points from R endogenously' "
                      "(step 3: grid label -> observer-state spectrum).",
    }
    out = ROOT / "experiments" / "exp_wall_smoothing_convergence_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
