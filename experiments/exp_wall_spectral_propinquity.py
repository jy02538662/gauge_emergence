"""第一块砖：谱邻近性的最粗糙版 —— Hausdorff 距离度量「离散谱 → 连续区间」收敛。

框架（方向 1 + 4 + 6）：
  1 弱谱三元组（不要紧 D）· 4 谱邻近性收敛（spectral propinquity）· 6 粗粒化公理化。
本块砖只做方向 4 的最粗糙版：Hausdorff 距离 d_H(σ_N, 连续区间) 度量谱点集的收敛。

精确性边界（诚实标注，用户要求）：
  (1) Hausdorff 距离 ≠ spectral propinquity —— 只比谱点集的几何距离，
      不比态空间（Hilbert）与代数（Lipschitz 结构）。这是「最粗糙版」。
  (2) 边缘收敛 ≠ 谱分布收敛 —— 只度量「点集填满区间」，不度量谱测度
      （归一化计数测度）的弱收敛（后者 = Weyl 律，是下一步）。
  (3) 不碰 P4（紧 D）—— 这里收敛到有界区间 [-2,2]，Toeplitz 算子有限有界，
      与「R 无紧算子 / Dixmier 迹发散」无关。P4 是第三步（弱谱三元组）的事。

主检验对象：对称三对角 Toeplitz 邻接矩阵 T_N（路径图 P_N 的邻接矩阵，次对角 1）：
  · 谱 σ_N = {2cos(kπ/(N+1)) : k=1..N} → 连续区间 [-2,2]。
  · 符号（sympy）：本征值精确 = 2cos(kπ/(N+1))；charpoly = U_N(λ/2)（第二类切比雪夫）；
    Hausdorff 距离 d_H 精确 = 2 sin(π/(2(N+1)))（N 偶）/ sin(π/(N+1))（N 奇）≈ π/N
    （一阶 O(1/N)，由 bulk 最大间隔决定，奇偶同阶）；
    边缘空隙精确 = 4 sin^2(π/(2(N+1))) ≈ π^2/N^2（二阶 O(1/N^2)，= 边缘间隔 = 尺度生成）。
  · 数值（numpy）：对角化 + 手写 d_H + 拟合收敛率 ≈ -1（一阶）。

对照：前向差分算子（周期）谱的逐点收敛 |λ_k − i2πk| = O(1/N)（呼应 exp_wall_smoothing_convergence），
  说明「Hausdorff 收敛（有界区间）」与「逐点收敛（发散谱）」是两种互补形态。

Code: `py -m experiments.exp_wall_spectral_propinquity`
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


def toeplitz_numpy(N):
    """N×N 对称三对角 Toeplitz（次对角 1，Dirichlet 路径图邻接矩阵）。"""
    return np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1)


def hausdorff_to_interval(ev, a=-2.0, b=2.0):
    """Hausdorff 距离 d_H(离散谱点集 ev, 连续区间 [a,b])。

    d_H = max(左边缘空隙, 右边缘空隙, 最大中间间隔的一半)。
    区间被升序谱点分成 N+1 段，每段的「覆盖空隙半径」的最大值即 d_H。
    """
    ev = np.sort(ev)
    left = ev[0] - a
    right = b - ev[-1]
    mid = (np.max(np.diff(ev)) / 2.0) if len(ev) > 1 else 0.0
    return max(left, right, mid)


def forward_diff_eigenvalues(N):
    """周期 [0,1]，步长 h=1/N，前向差分本征值（复数，收敛到 i2πk）。"""
    h = 1.0 / N
    k = np.arange(N)
    return (np.exp(2j * np.pi * k / N) - 1.0) / h


def sympy_section():
    """符号验证：本征值精确形式 + charpoly + Hausdorff 距离精确闭合形式。"""
    print("=== sympy 符号验证 ===")
    print()

    # (1) 本征值精确 = 2cos(kπ/(N+1))
    print("(1) 本征值精确 = 2cos(kπ/(N+1))")
    eig_exact = True
    for N in [3, 4, 5, 6, 7, 8]:
        T = sp.zeros(N, N)
        for i in range(N):
            for j in range(N):
                if abs(i - j) == 1:
                    T[i, j] = 1
        eigs = sorted([float(sp.re(sp.N(e, 40))) for e in T.eigenvals().keys()])
        expected = sorted([float(sp.N(2 * sp.cos(sp.pi * k / (N + 1)), 40)) for k in range(1, N + 1)])
        maxdev = max(abs(eigs[i] - expected[i]) for i in range(N))
        ok = maxdev < 1e-10
        eig_exact &= ok
        print(f"   N={N}: max|λ_num - 2cos(kπ/(N+1))| = {float(maxdev):.2e}  {'OK' if ok else 'FAIL'}")
    print()

    # (2) charpoly(T_N) = U_N(λ/2)
    print("(2) 特征多项式 = U_N(λ/2)（第二类切比雪夫）")
    lam = sp.symbols('lam')
    charpoly_ok = True
    for N in [3, 4, 5, 6, 7, 8]:
        T = sp.zeros(N, N)
        for i in range(N):
            for j in range(N):
                if abs(i - j) == 1:
                    T[i, j] = 1
        cp = sp.Poly(T.charpoly(lam), lam).as_expr()
        u = sp.expand(sp.chebyshevu(N, lam / 2))
        ok = sp.simplify(cp - u) == 0
        charpoly_ok &= ok
        print(f"   N={N}: charpoly == U_N(λ/2): {'OK' if ok else 'FAIL'}")
    print()

    # (3) Hausdorff 距离精确闭合形式 = 2 sin(π/(2(N+1)))
    print("(3) Hausdorff 距离 d_H(σ_N, [-2,2]) 精确闭合形式")
    nsym = sp.symbols('nsym', positive=True)
    dH_even = 2 * sp.sin(sp.pi / (2 * (nsym + 1)))            # N 偶：最大间隔在 k=N/2，sin=1
    dH_odd = sp.sin(sp.pi / (nsym + 1))                        # N 奇：max sin=cos，二倍角合并
    edge_formula = 4 * sp.sin(sp.pi / (2 * (nsym + 1))) ** 2   # 边缘空隙（不分奇偶）
    print("   精确形式：d_H = 2 sin(π/(2(N+1))) (N偶) / sin(π/(N+1)) (N奇)；均渐近 π/N")
    print("   边缘空隙 = 4 sin^2(π/(2(N+1))) ≈ π^2/N^2（不分奇偶）")
    dH_ok = True
    for N in [3, 4, 5, 6, 7, 8, 10, 16]:
        eigs = sorted([float(sp.N(2 * sp.cos(sp.pi * k / (N + 1)), 40)) for k in range(1, N + 1)])
        left = eigs[0] + 2
        right = 2 - eigs[-1]
        mid = max([(eigs[i + 1] - eigs[i]) / 2 for i in range(N - 1)])
        dH_num = max(left, right, mid)
        dH_formula = dH_even if N % 2 == 0 else dH_odd
        dH_formula_num = float(sp.N(dH_formula.subs(nsym, N), 40))
        ok = abs(dH_num - dH_formula_num) < 1e-12
        dH_ok &= ok
        print(f"   N={N}: d_H(直接)={dH_num:.6f}  公式(奇偶)={dH_formula_num:.6f}  {'OK' if ok else 'FAIL'}")
    # 渐近展开：d_H ~ π/N（一阶，奇偶同阶）；边缘空隙 ~ π^2/N^2（二阶）
    dH_even_series = sp.series(dH_even, nsym, sp.oo, 2)
    dH_odd_series = sp.series(dH_odd, nsym, sp.oo, 2)
    edge_series = sp.series(edge_formula, nsym, sp.oo, 3)
    print(f"   渐近展开 d_H(偶) ~ {dH_even_series}；d_H(奇) ~ {dH_odd_series}（一阶 O(1/N)，奇偶同阶）")
    print(f"   渐近展开 边缘空隙 ~ {edge_series}（二阶 O(1/N^2)，= 边缘间隔 = 尺度生成）")
    print()

    # (4) 对照：前向差分逐点收敛（符号验证本征值闭合形式）
    print("(4) 对照：前向差分本征值 λ_k = (e^{2πik/N}-1)/h → i2πk（逐点，一阶）")
    k, x = sp.symbols('k x', positive=True)
    lam_diff = (sp.exp(2 * sp.I * sp.pi * k * x) - 1) / x   # x = 1/n = h
    lam_series = sp.series(lam_diff, x, 0, 2)
    print(f"   差分本征值渐近（x=1/n→0，k 固定）：{sp.simplify(lam_series.removeO())} + O(x)")
    print()

    return {
        "eigenvalue_exact": bool(eig_exact),
        "charpoly_chebyshev": bool(charpoly_ok),
        "hausdorff_formula_exact": bool(dH_ok),
        "hausdorff_asymptotic": "d_H = 2 sin(pi/(2(N+1))) ~ pi/N (order 1)",
        "edge_gap_asymptotic": "4 sin^2(pi/(2(N+1))) ~ pi^2/N^2 (order 2)",
    }


def numpy_section():
    """数值验证：Hausdorff 距离收敛率 O(1/N)，边缘间隔收敛率 O(1/N^2)。"""
    print("=== numpy 数值验证 ===")
    print()

    Ns = [16, 32, 64, 128, 256, 512, 1024, 2048]
    dH_list, edge_list = [], []
    for N in Ns:
        T = toeplitz_numpy(N)
        ev = np.linalg.eigvalsh(T)          # 对称，升序
        dH = hausdorff_to_interval(ev, -2.0, 2.0)
        edge_gap = ev[-1] - ev[-2]          # 升序后最大两个的间隔 = 边缘间隔
        dH_list.append(dH)
        edge_list.append(edge_gap)

    slope_dH = np.polyfit(np.log(Ns), np.log(dH_list), 1)[0]
    slope_edge = np.polyfit(np.log(Ns), np.log(edge_list), 1)[0]
    print("   N        d_H(σ_N,[-2,2])      d_H*N        edge_gap      edge_gap*N^2")
    for i, N in enumerate(Ns):
        print(f"   {N:>5}    {dH_list[i]:.8f}    {dH_list[i]*N:6.4f}    "
              f"{edge_list[i]:.8f}    {edge_list[i]*N**2:6.3f}")
    print()
    print(f"   Hausdorff 距离收敛率 ~ N^{slope_dH:.2f}（期望 -1，一阶）")
    print(f"   边缘间隔收敛率     ~ N^{slope_edge:.2f}（期望 -2，二阶，= 尺度生成）")
    print(f"   d_H*N → π = {np.pi:.4f}（渐近验证，d_H*N 最后一列 = {dH_list[-1]*Ns[-1]:.4f}）")
    print()

    # 对照：前向差分逐点收敛
    print("   对照：前向差分 |λ_k - i2πk| 收敛率（逐点）")
    for k in [1, 2, 3]:
        errs = [abs(forward_diff_eigenvalues(N)[k] - 1j * 2 * np.pi * k) for N in Ns]
        s = np.polyfit(np.log(Ns), np.log(errs), 1)[0]
        print(f"      k={k}: ~ N^{s:.2f}（期望 -1）")
    print()

    return {
        "hausdorff_slope": float(slope_dH),
        "edge_gap_slope": float(slope_edge),
        "dH_times_N_last": float(dH_list[-1] * Ns[-1]),
    }


def main():
    print("=== 第一块砖：Hausdorff 距离度量离散谱 → 连续区间收敛 ===")
    print("（谱邻近性的最粗糙版；精确性边界见 docstring）")
    print()

    sym_res = sympy_section()
    num_res = numpy_section()

    print("=== 结论 ===")
    print("  1. Hausdorff 距离 d_H(σ_N, [-2,2]) = 2 sin(π/(2(N+1))) (N偶) / sin(π/(N+1)) (N奇) ~ π/N，一阶 O(1/N)。")
    print("     由 bulk 最大间隔决定（谱点集填满区间的「最坏空隙」）。")
    print("  2. 边缘间隔 = 4 sin^2(π/(2(N+1))) ~ π^2/N^2，二阶 O(1/N^2)，= 尺度生成（scale_chebyshev 同族）。")
    print("  3. 精确性边界：Hausdorff ≠ propinquity（不比态空间/代数）；边缘收敛 ≠ 谱分布收敛；不碰 P4。")
    print("  4. 对照：差分算子谱逐点收敛 O(1/N)（发散谱），与 Hausdorff 收敛（有界区间）互补。")
    print()

    summary = {
        "question": "does discrete spectrum converge to continuous interval, measured by Hausdorff distance "
                    "(coarsest version of spectral propinquity, direction 4)?",
        "object": "symmetric tridiagonal Toeplitz adjacency matrix T_N (path graph P_N)",
        "spectrum": "sigma_N = {2cos(k pi/(N+1))} -> [-2,2]",
        "sympy": sym_res,
        "numpy": num_res,
        "precision_boundaries": [
            "Hausdorff distance != spectral propinquity (no state space / algebra)",
            "edge convergence != spectral measure convergence (no Weyl law yet)",
            "does NOT touch P4 (compact D) - that is step 3 (weak spectral triple)",
        ],
        "conclusion": "d_H(sigma_N, [-2,2]) = 2 sin(pi/(2(N+1))) (N even) / sin(pi/(N+1)) (N odd) ~ pi/N (order 1); "
                      "edge gap ~ pi^2/N^2 (order 2) = scale generation. "
                      "First symbolic+numeric anchor of the smoothing line.",
    }
    out = ROOT / "experiments" / "exp_wall_spectral_propinquity_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
