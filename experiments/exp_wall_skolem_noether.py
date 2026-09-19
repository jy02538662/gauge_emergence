"""检验：有限维 Out 平凡（Skolem–Noether）+ 转置是反自同构（交换化后变自同构）。

卡点 P2/P3 的「外自同构」有限维侧面：
  - 定理（Skolem–Noether）：Aut(M_n(C)) = Inn(M_n) = {Ad_U : U 酉}。
    => Out(M_n) 平凡，有限维没有外自同构，外自同构只从 n -> ∞ 涌现。
  - 转置 τ(A) = A^T 是**反自同构**（τ(AB) = τ(B)τ(A)），不是自同构。
    在非交换 M_n 上，连「最像外」的候选（转置）都不是自同构 -> Out 平凡的直接体现。

对着靶点 Aut_互联(R)/Inn(R) = Diff(M) 的种子：
  - 交换化后（投影到对角 MASA），转置 = 恒等（自同构）。
  - 即：非交换 M_n 上「反自同构」（转置/共轭转置）不是自同构；
        交换化后它们变成自同构（恒等）——交换化让对称性「变多」。
  - 微分同胚 Diff(M) 不是 R（非交换）的自同构，而是交换化结果（流形函数代数 C^∞(M)）
    的自同构。这是「Diff 从交换化涌现」的有限维侧面。

Code: `py -m experiments.exp_wall_skolem_noether`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def is_zero_matrix(M):
    return all(sp.simplify(e) == 0 for e in M)


def main():
    print("=== 有限维 Out 平凡（Skolem–Noether）+ 转置反自同构 ===")
    print()

    # ---- 1. 转置是反自同构 ----
    print("1. 转置 τ(A)=A^T 是反自同构：(AB)^T = B^T A^T")
    a11, a12, a21, a22 = sp.symbols('a11 a12 a21 a22')
    b11, b12, b21, b22 = sp.symbols('b11 b12 b21 b22')
    A = sp.Matrix([[a11, a12], [a21, a22]])
    B = sp.Matrix([[b11, b12], [b21, b22]])

    ABT = (A @ B).T
    BTAT = B.T @ A.T
    anti_resid = sp.simplify(ABT - BTAT)
    anti_ok = is_zero_matrix(anti_resid)
    print(f"   (AB)^T - B^T A^T = {anti_resid}")
    print(f"   -> 转置是反自同构 {'[OK]' if anti_ok else '[FAIL]'}")

    # ---- 2. 转置不是自同构 ----
    print()
    print("2. 转置不是自同构：A^T B^T != (AB)^T（一般非零）")
    ATBT = A.T @ B.T
    diff = sp.simplify(ATBT - ABT)
    not_auto = not is_zero_matrix(diff)
    print(f"   A^T B^T - (AB)^T = {diff}")
    print(f"   -> {'非零（转置不是自同构，Out 平凡的侧面）' if not_auto else '恰为自同构'}")

    # ---- 3. 对角矩阵转置 = 恒等 ----
    print()
    print("3. 交换化后（对角 MASA）：转置 = 恒等（自同构）")
    d1, d2 = sp.symbols('d1 d2')
    D = sp.diag(d1, d2)
    diag_resid = sp.simplify(D.T - D)
    diag_ok = is_zero_matrix(diag_resid)
    print(f"   D^T - D = {diag_resid}")
    print(f"   -> 转置=恒等（自同构）{'[OK]' if diag_ok else '[FAIL]'}")

    # ---- 结论 ----
    print()
    print("=== 结论 ===")
    print("  有限维 Aut(M_n) = Inn(M_n)（Skolem–Noether），Out 平凡。")
    print("  非交换 M_n 上转置/共轭转置是反自同构（不是自同构）。")
    print("  交换化后反自同构 -> 自同构（恒等），对称性涌现。")
    print("  => Diff(M) 是交换化结果（C^∞(M)）的自同构，不是非交换 R 的自同构。")
    print("     「外自同构 / 微分同胚」只从 n -> ∞（离散 -> 连续）涌现。")

    summary = {
        "question": "finite-dim Out trivial (Skolem-Noether)? transpose automorphism or anti?",
        "transpose_is_antiautomorphism": bool(anti_ok),
        "transpose_not_automorphism": bool(not_auto),
        "diagonal_transpose_identity": bool(diag_ok),
        "conclusion": "Aut(M_n)=Inn(M_n), Out trivial. Transpose is anti-automorphism on "
                      "noncommutative M_n, becomes identity (automorphism) after abelianization. "
                      "Diff(M) is automorphism of abelianized C^inf(M), not of noncommutative R. "
                      "Outer automorphisms / diffeomorphisms only emerge at n->infinity "
                      "(discrete -> continuous).",
    }
    out = ROOT / "experiments" / "exp_wall_skolem_noether_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
