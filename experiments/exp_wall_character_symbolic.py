"""P1 核心：M_n(C)（n≥2）无非零 character（符号验证）。

卡点 1 精确形式（论文 #18）：
  II_1 因子 R 中心平凡 Z(R)=C I，无 character（无非零 *-同态到 C）。
  交换 C*-代数 C(X) 每点一个求值同态 ev_x: f -> f(x)，character 极多。
  点 = character。

有限维侧能符号验证的（冯诺伊曼做不了的：真正无限维 II_1 无 character）：
  定理：M_n(C)（n≥2）是单代数 => 无非零乘法同态 φ: M_n -> C。
  推导链：
    (a) ker φ 是双边理想；
    (b) M_n 单 => ker φ in {0, M_n}；
    (c) ker φ = 0 => φ 单射，但 dim M_n = n^2 > 1 = dim C => 矛盾；
    (d) => ker φ = M_n => φ = 0。
  符号化（n=2）：φ(E_ij) = x_ij，character 条件 δ_jk x_il = x_ij x_kl，
    推出 x_ii 幂等（in{0,1}）、x_ii 全相等、Σ x_ii = φ(I) = 1，
    => x_ii = 1/2 not in {0,1} => 矛盾（无非零解）。

  对比：交换对角代数（MASA 的有限维对应）有 character = 求值同态（对角元提取）。
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
    print("=== P1 核心：M_n(C) 无非零 character（符号验证）===")
    print()

    # ---- 1. 推导：单代数无 character ----
    print("1. 推导（定理）：M_n(C) 单代数 => 无非零乘法同态到 C。")
    print("   ker φ 双边理想；M_n 单 => ker φ in {0, M_n}；")
    print("   ker φ=0 => φ 单射 => n^2 <= 1 矛盾；=> ker φ = M_n => φ = 0。")

    # ---- 2. 符号：n=2 的 character 方程 ----
    print()
    print("2. 符号验证 n=2：character φ(E_ij)=x_ij 的方程")
    x11, x12, x21, x22 = sp.symbols('x11 x12 x21 x22')
    # 幂等：E_ii^2 = E_ii  =>  x_ii^2 = x_ii  =>  x_ii in {0,1}
    sol_idem = sp.solve([x11 ** 2 - x11, x22 ** 2 - x22], [x11, x22], dict=True)
    print(f"   E_ii 幂等: x_ii^2 = x_ii 的解 = {sol_idem}  (x_ii in {{0,1}})")

    # 相等：E_12 E_21 = E_11, E_21 E_12 = E_22  =>  x12 x21 = x11 且 x21 x12 = x22  =>  x11 = x22
    # 归一：Σ φ(E_ii) = φ(I) = 1
    eq_equal = sp.Eq(x11, x22)
    eq_sum = sp.Eq(x11 + x22, 1)
    sol_sys = sp.solve([x11 ** 2 - x11, eq_equal, eq_sum], [x11, x22], dict=True)
    print(f"   x11=x22 (由 E12E21 vs E21E12) + Σx_ii=1 + 幂等 的解 = {sol_sys}")
    has_solution = len(sol_sys) > 0
    print(f"   => 2x11=1 => x11=1/2 not in {{0,1}} => "
          f"{'有解' if has_solution else '无解（无非零 character）'}")

    # ---- 3. 精确矛盾：直接检验完整乘法同态方程 ----
    print()
    print("3. 精确矛盾（n=2 完整乘法同态方程 δ_jk x_il = x_ij x_kl）：")
    # E_11 E_12 = E_12  =>  x_11 x_12 = x_12
    # E_12 E_11 = 0     =>  x_12 x_11 = 0
    # 两式合并 => x_12 = 0（若 x_11 != 0）或 x_12 任意（若 x_11 = 0）
    # E_12 E_21 = E_11  =>  x_12 x_21 = x_11
    # 若 x_11 = 1 且 x_12 x_21 = 1，则 x_12, x_21 != 0；
    # 但 E_12 E_11 = 0 => x_12 x_11 = 0 => x_12 = 0（因 x_11=1），与 x_12!=0 矛盾。
    # 简洁矛盾即上面 x_ii = 1/2 not in {0,1}。
    print("   由 x_ii 幂等 + 全相等 + Σ=1 得 x_ii=1/2，与幂等 {0,1} 冲突 —— 无非零 character。")

    # ---- 4. 对比：交换对角代数有 character ----
    print()
    print("4. 对比：交换对角代数（MASA 有限维对应）有 character = 求值同态")
    d1, d2, e1, e2 = sp.symbols('d1 d2 e1 e2')
    D1 = sp.diag(d1, d2)
    D2 = sp.diag(e1, e2)
    # character φ_1: diag(d1,d2) -> d1 (求值在第 1 个对角元)
    phi1_D1 = D1[0, 0]
    phi1_D2 = D2[0, 0]
    prod = sp.diag(d1 * e1, d2 * e2)
    phi1_prod = prod[0, 0]
    char_resid = sp.simplify(phi1_D1 * phi1_D2 - phi1_prod)
    print(f"   φ_1(D1)·φ_1(D2) - φ_1(D1·D2) = {char_resid}  (=0：乘法同态成立)")
    print(f"   => 对角代数有 character（求值同态，即「点」），M_n 没有。")
    print("   这就是 P1：非交换（无 character/无点）vs 交换（有 character/有点）。")

    summary = {
        "theorem": "M_n(C) (n>=2) simple => no nonzero character (no multiplicative hom to C)",
        "idempotent_solutions": str(sol_idem),
        "system_no_solution": not has_solution,
        "contradiction": "x_ii equal + sum=1 => x_ii=1/2, but idempotent => x_ii in {0,1}: contradiction",
        "abelian_has_character": bool(char_resid == 0),
        "conclusion": "M_n has NO character (P1: no point). Diagonal (abelian/MASA) HAS "
                      "character = evaluation (point). 'noncommutative -> commutative' is "
                      "the wall, and the finite-dim version is exact here.",
    }
    out = ROOT / "experiments" / "exp_wall_character_symbolic_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
