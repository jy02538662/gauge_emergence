"""第三块砖（②b）：Lipschitz 半范的桥梁传递 —— propinquity 的「reach」成分。

框架（方向 4 谱邻近性收敛）②b：① 谱点集 Hausdorff、②a 谱测度 Wasserstein（态空间），
本块砖上到「代数层」：直接比 Lipschitz 半范 L(a) = ||[D, a]|| 通过「采样桥梁」的传递。
这是 Latrémolière propinquity 里「bridge reach」的 1D 交换特例。

核心命题（可符号/数值验证）：
  离散 Lipschitz 半范  L_N(S_N(f)) = ||[T_N, diag(f(λ_k))]||   （差分）
  连续 Lipschitz 半范  L_∞(f) = ||f'||_∞                        （导数）
  桥梁（采样 S_N: f ↦ (f(λ_1),...,f(λ_N))）把导数变差分，缩放因子 = 谱间隔 δλ(λ)：
      L_N(S_N(f)) = 2 · sup_λ |f'(λ)| · δλ(λ) · (1 + O(1/N))
  其中谱间隔 δλ(λ) = 1/(N·ρ(λ)) = π√(4-λ^2)/N（arcsine 密度 ρ=1/(π√(4-λ^2)) 的倒数）。

关键洞察（接第二块砖）：缩放因子 δλ(λ) 是**位置依赖**的（bulk 稀疏 ~2π/N、边缘密 ~π^2/N^2），
正是 arcsine 非均匀密度给的。所以「差分→导数」的正确缩放不是常数，而是 δλ(λ) = 1/(N·ρ(λ))。
这坐实第二块砖的 arcsine 发现是桥梁传递的「正确缩放」。

精确性边界（诚实标注）：
  (1) 这是「自然桥梁（采样/插值）」的 reach，不是 propinquity 的完整定义
      （完整定义要 inf over all bridges 的 height+reach，且要两个代数无共同环境）。
  (2) 采样桥梁预设了「σ_N ⊂ [-2,2]」这个嵌入——真正摆脱共同环境要 Gromov-Hausdorff 桥梁（②b 后半）。
  (3) 不碰 P4（紧 D）。

Code: `py -m experiments.exp_wall_lipschitz_bridge`
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


def discrete_lipschitz(f_vals):
    """L_N(S_N(f)) = ||[T_N, diag(f(λ_k))]||：反对称三对角（次对角 = 差分）的谱范数。"""
    c = np.diff(f_vals)              # a_{i+1} - a_i（升序谱点）
    A = np.diag(c, 1) - np.diag(c, -1)
    return float(np.linalg.norm(A, 2))


def sup_fp_times_gap(fprime, M=200001):
    """sup_λ |f'(λ)|·√(4-λ^2)（= (N/π)·L_N 的极限，数值最大化）。"""
    xs = np.linspace(-2.0, 2.0, M)
    return float(np.max(np.abs(fprime(xs)) * np.sqrt(4.0 - xs ** 2)))


def sympy_section():
    print("=== sympy 符号验证 ===")
    print()

    # (1) 差分结构：[T_N, diag(a)] 是反对称三对角，次对角 = a_{i+1}-a_i
    N = sp.Integer(4)
    a = sp.symbols('a0:4')
    T = sp.zeros(N, N)
    for i in range(N):
        for j in range(N):
            if abs(i - j) == 1:
                T[i, j] = 1
    D = sp.Matrix(N, N, lambda i, j: a[i] if i == j else 0)
    comm = T * D - D * T
    # 验证 (i,i+1) 元 = a_{i+1}-a_i，对角 = 0，(i,i+1) 与 (i+1,i) 反对称
    offdiag_ok = all(sp.simplify(comm[i, i + 1] - (a[i + 1] - a[i])) == 0 for i in range(N - 1))
    diag_ok = all(sp.simplify(comm[i, i]) == 0 for i in range(N))
    antisym_ok = all(sp.simplify(comm[i, j] + comm[j, i]) == 0 for i in range(N) for j in range(i + 1, N))
    print(f"(1) [T_N, diag(a)] 反对称三对角、次对角 = 差分："
          f"次对角 {'OK' if offdiag_ok else 'FAIL'} / 对角 {'OK' if diag_ok else 'FAIL'} / "
          f"反对称 {'OK' if antisym_ok else 'FAIL'}")

    # (2) 谱间隔 δλ(λ) = 1/(N·ρ(λ)) = π√(4-λ^2)/N（arcsine 密度倒数）
    Nsym, lam = sp.symbols('Nsym lam', positive=True)
    rho = 1 / (sp.pi * sp.sqrt(4 - lam ** 2))
    gap = sp.simplify(1 / (Nsym * rho))
    gap_bulk = sp.simplify(gap.subs(lam, 0))
    gap_ok = sp.simplify(gap - sp.pi * sp.sqrt(4 - lam ** 2) / Nsym) == 0
    bulk_ok = sp.simplify(gap_bulk - 2 * sp.pi / Nsym) == 0
    print(f"(2) 谱间隔 δλ(λ) = π√(4-λ^2)/N，bulk λ=0 处 = 2π/N："
          f"公式 {'OK' if gap_ok else 'FAIL'} / bulk {'OK' if bulk_ok else 'FAIL'}")

    # (3) 因子 2：反对称三对角 Toeplitz（次对角 1）谱范数 = 2cos(π/(N+1)) ≈ 2
    factor2_ok = True
    for N2 in [3, 4, 5, 6, 8, 16, 32, 64]:
        A = np.diag(np.ones(N2 - 1), 1) - np.diag(np.ones(N2 - 1), -1)
        normA = float(np.linalg.norm(A, 2))
        expected = 2.0 * np.cos(np.pi / (N2 + 1))
        ok = abs(normA - expected) < 1e-8
        factor2_ok &= ok
    print(f"(3) 因子 2：反对称三对角 Toeplitz 谱范数 = 2cos(π/(N+1)) ≈ 2（N=3..64）"
          f"{'OK' if factor2_ok else 'FAIL'}")

    print()
    print("   结论：Lipschitz 半范的桥梁传递 = 2 × 导数 × 非均匀谱间隔 δλ(λ)（arcsine 密度倒数）。")
    print()
    return {"commutator_structure": bool(offdiag_ok and diag_ok and antisym_ok),
            "spectral_gap_formula": bool(gap_ok and bulk_ok),
            "factor_2": bool(factor2_ok)}


def numpy_section():
    print("=== numpy 数值验证（Lipschitz 桥梁传递收敛）===")
    print()

    Ns = [16, 32, 64, 128, 256, 512, 1024, 2048]
    tests = [
        ("f(λ)=λ", lambda x: x, lambda x: np.ones_like(x), 2.0),
        ("f(λ)=λ^2/2", lambda x: x ** 2 / 2, lambda x: x, 2.0),
        ("f(λ)=sin λ", np.sin, np.cos, 2.0),
        ("f(λ)=cos λ", np.cos, lambda x: -np.sin(x), None),
    ]
    print("   （L_N · N/(2π) 应 → sup_λ |f'(λ)|√(4-λ^2)）")
    print()
    results = {}
    for name, f, fprime, expect in tests:
        sup_target = sup_fp_times_gap(fprime) if expect is None else expect
        ratios = []
        for N in Ns:
            ev = spectrum(N)
            L_N = discrete_lipschitz(f(ev))
            ratios.append(L_N * N / (2 * np.pi))
        slope = np.polyfit(np.log(Ns), np.log(np.abs(np.array(ratios) - sup_target) + 1e-18), 1)[0]
        results[name] = {"sup_target": sup_target, "last_ratio": ratios[-1], "slope": float(slope)}
        print(f"   {name}: sup|f'|√(4-λ^2) = {sup_target:.6f}，L_N·N/(2π) 最后 = {ratios[-1]:.6f}，"
              f"误差收敛 ~ N^{slope:.2f}（期望 -1）")
    print()

    return results


def main():
    print("=== 第三块砖（②b）：Lipschitz 半范的桥梁传递（propinquity 的 reach 成分）===")
    print("（精确性边界见 docstring）")
    print()

    sym_res = sympy_section()
    num_res = numpy_section()

    print("=== 结论 ===")
    print("  1. 离散 Lipschitz 半范 L_N(f) = ||[T_N, diag(f(λ))]|| = 2·sup_λ |f'(λ)|·δλ(λ)·(1+O(1/N))。")
    print("  2. 缩放因子 δλ(λ) = π√(4-λ^2)/N 是位置依赖的（= arcsine 密度倒数），非均匀采样。")
    print("  3. 精确性边界：这是「采样桥梁」的 reach，非完整 propinquity（缺 inf over all bridges / 无共同环境）。")
    print("  4. 第二块砖的 arcsine 发现 = 桥梁传递的「正确缩放」，两砖串起来了。")
    print()

    summary = {
        "question": "does discrete Lipschitz seminorm L_N(f)=||[T_N,diag(f(λ))]|| transfer to continuous "
                    "through the sampling bridge, with position-dependent scale δλ(λ)=1/(N·ρ(λ))?",
        "proposition": "L_N(S_N(f)) = 2·sup_λ |f'(λ)|·δλ(λ)·(1+O(1/N)), δλ(λ)=π√(4-λ^2)/N",
        "sympy": sym_res,
        "numpy": {k: {kk: (vv if not isinstance(vv, float) or True else vv) for kk, vv in v.items()} for k, v in num_res.items()},
        "precision_boundaries": [
            "sampling bridge reach, not full propinquity (no inf over all bridges / no common-env-free)",
            "sampling bridge presupposes σ_N ⊂ [-2,2] embedding; Gromov-Hausdorff bridge is ②b second half",
            "does NOT touch P4 (compact D)",
        ],
        "conclusion": "Lipschitz bridge transfer = 2 × derivative × position-dependent spectral gap δλ(λ). "
                      "The arcsine discovery (②a) IS the correct scale for the bridge.",
    }
    out = ROOT / "experiments" / "exp_wall_lipschitz_bridge_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
