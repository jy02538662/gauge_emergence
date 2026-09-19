"""符号推：M_3(C) 上 Z_3（循环移位）的不变子代数是否交换。

Z_3 生成元 sigma = 循环移位（sigma^3 = I）。
不变子代数 R^sigma = {a : sigma a sigma^{-1} = a} = {a : sigma a = a sigma}。
验证：R^sigma 是否交换，Gelfand 谱是什么。

Code: `py -m experiments.exp_verify_Z3`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    print("=== 符号推：M_3(C) 上 Z_3 不变子代数 ===")
    print()

    sigma = sp.Matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    print(f"1. sigma = 循环移位, sigma^3 = {sp.simplify(sigma ** 3).tolist()}  (应 = I)")

    a11, a12, a13, a21, a22, a23, a31, a32, a33 = sp.symbols(
        'a11 a12 a13 a21 a22 a23 a31 a32 a33')
    a = sp.Matrix([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
    diff = sp.simplify(sigma @ a - a @ sigma)
    sol = sp.solve(list(diff), [a11, a12, a13, a21, a22, a23, a31, a32, a33],
                   dict=True)
    print()
    print("2. 解 sigma a = a sigma 得不变子代数形式：")
    if sol:
        s = sol[0]
        b = sp.Matrix([[s.get(a11, 0), s.get(a12, 0), s.get(a13, 0)],
                       [s.get(a21, 0), s.get(a22, 0), s.get(a23, 0)],
                       [s.get(a31, 0), s.get(a32, 0), s.get(a33, 0)]])
        print(f"   a = {b.tolist()}")
        print("   => 循环矩阵（circulant），形如 [[d,p,q],[q,d,p],[p,q,d]]，3 参数（d,p,q）")
    else:
        print("   （无法符号解，用已知结果：循环矩阵）")

    # ---- 验证循环矩阵交换 ----
    d, p, q, d2, p2, q2 = sp.symbols('d p q d2 p2 q2')
    C1 = sp.Matrix([[d, p, q], [q, d, p], [p, q, d]])
    C2 = sp.Matrix([[d2, p2, q2], [q2, d2, p2], [p2, q2, d2]])
    comm = sp.simplify(C1 @ C2 - C2 @ C1)
    print()
    print(f"3. 两个循环矩阵的交换子 [C1, C2] = {comm.tolist()}")
    print(f"   => 循环矩阵交换：{comm == sp.zeros(3, 3)}")

    # ---- Gelfand 谱 = sigma 的本征值（3 个点）----
    ev = sigma.eigenvals()
    print()
    print(f"4. R^sigma 的 Gelfand 谱 = sigma 的本征值 = {ev}")
    print("   => 3 个点（1, omega, omega^2），是离散的 3 点空间，不是连续流形。")

    print()
    print("=== 结论 ===")
    print("  有限维 M_3(C)：Z_3 不变子代数 = 循环矩阵 = 交换（3 个点）。✅ 用户直觉对。")
    print("  但两个问题：")
    print("  1. 无限维 II_1 因子：sigma 的谱可能连续（重数），中心化子 R^sigma 可能")
    print("     非交换（只在谱无重数时交换）。")
    print("  2. 即使交换，R^sigma 的谱 = 3 个离散点，不是连续流形（物理空间需要）。")
    print("     「3 个点 -> 连续流形」= 离散 -> 连续（老墙）。")

    summary = {
        "sigma3": str(sp.simplify(sigma ** 3)),
        "circulant_commutator_zero": bool(comm == sp.zeros(3, 3)),
        "gelfand_spectrum": "3 points (1, omega, omega^2)",
        "conclusion": "finite-dim M_3: Z_3-fixed subalgebra = circulant = abelian (3 points). "
                      "But (1) infinite II_1 may have spectral multiplicity -> centralizer "
                      "non-abelian; (2) even if abelian, spectrum = 3 discrete points, not a "
                      "continuous manifold. '3 points -> manifold' = discrete -> continuous (wall).",
    }
    out = ROOT / "experiments" / "exp_verify_Z3_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
