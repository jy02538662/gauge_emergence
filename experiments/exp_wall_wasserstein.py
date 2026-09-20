"""第二块砖：谱测度的 Wasserstein-1 收敛 —— spectral propinquity 的「态空间成分」。

框架（方向 4 谱邻近性收敛）第二步。第一块砖（Hausdorff 距离）只比「谱点集支撑」；
本块砖带上「态空间（概率测度）+ Lipschitz 对偶结构」——W_1 = sup_{f: Lip(f)<=1} |∫f dμ - ∫f dν|。

对象：离散经验谱测度 μ_N = (1/N) Σ_k δ_{2cos(kπ/(N+1))}（路径图邻接矩阵 T_N 的谱测度）
  vs 连续极限 μ_∞ = arcsine 分布 on [-2,2]，密度 1/(π√(4-λ^2))。
  W_1(μ_N, μ_∞) 度量谱测度的弱收敛 = 谱分布收敛（Weyl 律的 1D 版）。

关键纠正（比第一块砖更实质）：μ_N 的极限不是均匀分布，而是 arcsine 分布
（密度 1/(π√(4-λ^2))，在 ±2 处发散）——因为本征值 λ=2cos(kπ/(N+1)) 是 cos 的均匀采样，
推前测度 = arcsine，不是均匀。早先「谱分布收敛 = 均匀分布」是错的。
脚本用「vs arcsine → 0」和「vs uniform → 非零常数」两个 W_1 对照坐实这一点。

精确性边界（诚实标注）：
  (1) Wasserstein-1 ≠ spectral propinquity：W_1 度量「同一环境空间 [-2,2] 上的两个测度」，
      Latrémolière 的 propinquity 是 Gromov-Hausdorff 意义上的（两空间无需共同环境，用「桥梁」耦合）。
      本块砖是 propinquity 的「态空间成分」，缺「桥梁 / 代数传递」。
  (2) 带上态空间 + Lipschitz 对偶：W_1 的对偶形式 = MK 度量（propinquity 定义里态空间度量的经典情形）。
  (3) 不碰 P4（紧 D）：μ_∞ 是有界区间 [-2,2] 上的测度，与「R 无紧算子」无关。

Code: `py -m experiments.exp_wall_wasserstein`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def spectrum(N):
    """路径图 P_N 邻接矩阵的精确谱（升序）。"""
    return np.sort([2.0 * np.cos(np.pi * k / (N + 1)) for k in range(1, N + 1)])


def wasserstein1_cdf(ev, cdf, M=1_000_000):
    """W_1(μ_N, ν) = ∫|F_N - F_ν| dλ（1D，用 CDF 的 L¹ 距离）。

    ev: 升序谱点（μ_N 的支撑）；cdf: 目标测度 ν 的 CDF 函数。
    """
    ev = np.sort(ev)
    N = len(ev)
    xs = np.linspace(-2.0, 2.0, M)
    F_N = np.searchsorted(ev, xs, side="right") / N
    return np.trapz(np.abs(F_N - cdf(xs)), xs)


def cdf_arcsine(x):
    return 0.5 + np.arcsin(x / 2.0) / np.pi


def cdf_uniform(x):
    return (x + 2.0) / 4.0


def hausdorff_to_interval(ev, a=-2.0, b=2.0):
    ev = np.sort(ev)
    left = ev[0] - a
    right = b - ev[-1]
    mid = (np.max(np.diff(ev)) / 2.0) if len(ev) > 1 else 0.0
    return max(left, right, mid)


def sympy_section():
    print("=== sympy 符号验证（arcsine 分布是 μ_N 的极限）===")
    print()

    t, u, lam = sp.symbols('t u lam', real=True)
    rho = 1 / (sp.pi * sp.sqrt(4 - t ** 2))

    # (1) 归一化
    norm = sp.integrate(rho, (t, -2, 2))
    norm_ok = sp.simplify(norm - 1) == 0
    print(f"(1) ∫ 1/(π√(4-λ^2)) dλ on [-2,2] = {norm}  {'OK' if norm_ok else 'FAIL'}")

    # (2) CDF = 1/2 + arcsin(λ/2)/π
    cdf = sp.integrate(rho, (t, -2, u))
    cdf_expected = sp.Rational(1, 2) + sp.asin(u / 2) / sp.pi
    cdf_ok = sp.simplify(cdf - cdf_expected) == 0
    print(f"(2) CDF = {sp.simplify(cdf)}  {'OK' if cdf_ok else 'FAIL'}")

    # (3) 推前密度：λ=2cos(θ), θ~Uniform[0,π] ⟹ 密度 = 1/(π√(4-λ^2))
    #     ρ(λ) = (1/π)/|dλ/dθ| = 1/(2π sin θ) = 1/(2π√(1-λ^2/4)) = 1/(π√(4-λ^2))
    density = 1 / (2 * sp.pi * sp.sqrt(1 - (lam / 2) ** 2))
    target = 1 / (sp.pi * sp.sqrt(4 - lam ** 2))
    push_ok = sp.simplify(density - target) == 0
    print(f"(3) 推前密度 1/(2π√(1-λ^2/4)) == 1/(π√(4-λ^2))  {'OK' if push_ok else 'FAIL'}")

    print()
    print("   结论：μ_N = (1/N)Σδ_{2cos(kπ/(N+1))} 的连续极限 = arcsine 分布（非均匀）。")
    print()
    return {
        "arcsine_normalized": bool(norm_ok),
        "arcsine_cdf": bool(cdf_ok),
        "pushforward_density": bool(push_ok),
    }


def numpy_section():
    print("=== numpy 数值验证（W_1 收敛率 + 对照）===")
    print()

    Ns = [16, 32, 64, 128, 256, 512, 1024, 2048]
    w1_arc_list, w1_uni_list, dH_list = [], [], []
    for N in Ns:
        ev = spectrum(N)
        w1_arc_list.append(wasserstein1_cdf(ev, cdf_arcsine))
        w1_uni_list.append(wasserstein1_cdf(ev, cdf_uniform))
        dH_list.append(hausdorff_to_interval(ev))

    slope_arc = np.polyfit(np.log(Ns), np.log(w1_arc_list), 1)[0]
    slope_dH = np.polyfit(np.log(Ns), np.log(dH_list), 1)[0]

    print("   N       W_1(μ_N, arcsine)   W_1(μ_N, uniform)   d_H(σ_N, [-2,2])")
    for i, N in enumerate(Ns):
        print(f"   {N:>5}    {w1_arc_list[i]:.6e}      {w1_uni_list[i]:.6e}      {dH_list[i]:.6e}")
    print()
    print(f"   W_1(μ_N, arcsine) 收敛率 ~ N^{slope_arc:.2f}（期望 -1，一阶，→0）")
    print(f"   W_1(μ_N, uniform) 最后值 = {w1_uni_list[-1]:.4f}（非零常数：μ_N 不收敛到均匀）")
    print(f"   d_H(σ_N, [-2,2]) 收敛率 ~ N^{slope_dH:.2f}（对照，一阶，→0）")
    print()
    print("   对照结论：W_1 vs arcsine → 0、vs uniform → 常数，坐实「极限是 arcsine 不是均匀」。")
    print()

    return {
        "wasserstein_arcsine_slope": float(slope_arc),
        "hausdorff_slope": float(slope_dH),
        "wasserstein_uniform_last": float(w1_uni_list[-1]),
    }


def main():
    print("=== 第二块砖：谱测度的 Wasserstein-1 收敛（propinquity 的态空间成分）===")
    print("（精确性边界见 docstring）")
    print()

    sym_res = sympy_section()
    num_res = numpy_section()

    print("=== 结论 ===")
    print("  1. μ_N = (1/N)Σδ_{2cos(kπ/(N+1))} 弱收敛到 arcsine 分布（密度 1/(π√(4-λ^2))），非均匀。")
    print("  2. W_1(μ_N, arcsine) ~ N^-1（一阶，→0），与 Hausdorff 距离同阶，但度量谱测度（分布）而非谱点集（支撑）。")
    print("  3. 精确性边界：Wasserstein ≠ propinquity（缺 Gromov-Hausdorff 桥梁/代数传递）；不碰 P4。")
    print("  4. 纠正：谱分布收敛（Weyl 律 1D 版）= arcsine 分布，不是均匀分布。")
    print()

    summary = {
        "question": "does empirical spectral measure mu_N = (1/N) sum delta_{2cos(k pi/(N+1))} "
                    "converge weakly (Wasserstein-1) to arcsine, and at what rate?",
        "limit_measure": "arcsine on [-2,2], density 1/(pi sqrt(4-lambda^2)) (NOT uniform)",
        "sympy": sym_res,
        "numpy": num_res,
        "precision_boundaries": [
            "Wasserstein-1 != spectral propinquity (missing Gromov-Hausdorff bridge / algebra transfer)",
            "carries state space + Lipschitz dual (MK metric) - the state-space component of propinquity",
            "does NOT touch P4 (compact D)",
        ],
        "correction": "spectral distribution limit (1D Weyl) = arcsine, NOT uniform",
        "conclusion": "W_1(mu_N, arcsine) ~ N^-1 (order 1, -> 0); W_1(mu_N, uniform) -> nonzero constant. "
                      "State-space component of propinquity anchored.",
    }
    out = ROOT / "experiments" / "exp_wall_wasserstein_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
