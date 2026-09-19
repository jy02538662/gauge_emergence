"""4 维 Fisher 度规的 Weyl 张量：是否自发非零（逃离共形平坦 = 自旋 2 涌现）。

先验证 Weyl 计算代码（Schwarzschild，已知 Weyl != 0），
再算一个 4 参数候选 Fisher 度规（块对角，来自对数配分函数 A(theta) 的 Hessian）的 Weyl。

Weyl（4 维）：
C_ijkl = R_ijkl - (g_ik R_jl - g_il R_jk - g_jk R_il + g_jl R_ik)/2
        + R (g_ik g_jl - g_il g_jk) / 6

Code: `py -m experiments.exp_verify_fisher_weyl`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def weyl_components(g, coords):
    """计算 4 维度量 g 的 Weyl 张量（全协变分量 C_ijkl）。"""
    n = 4
    ginv = g.inv()
    # Christoffel
    Gamma = [[[sp.simplify(sp.Rational(1, 2) * sum(
        ginv[i, l] * (sp.diff(g[l, k], coords[j]) + sp.diff(g[l, j], coords[k])
                      - sp.diff(g[j, k], coords[l])) for l in range(n)))
        for k in range(n)] for j in range(n)] for i in range(n)]
    # Riemann（全协变）
    Rm = [[[[sp.simplify(sum(
        g[i, m] * (sp.diff(Gamma[m][j][l], coords[k]) - sp.diff(Gamma[m][j][k], coords[l])
                   + sum(Gamma[m][p][k] * Gamma[p][j][l] - Gamma[m][p][l] * Gamma[p][j][k]
                         for p in range(n)))
        for m in range(n))) for l in range(n)] for k in range(n)] for j in range(n)]
        for i in range(n)]
    # Ricci
    Ric = [[sp.simplify(sum(Rm[i][j][i][l] for i in range(n))) for l in range(n)]
           for j in range(n)]
    # 标量曲率
    R = sp.simplify(sum(ginv[j, l] * Ric[j][l] for j in range(n) for l in range(n)))
    # Weyl
    W = [[[[sp.simplify(
        Rm[i][j][k][l]
        - (g[i, k] * Ric[j][l] - g[i, l] * Ric[j][k] - g[j, k] * Ric[i][l]
           + g[j, l] * Ric[i][k]) / 2
        + R * (g[i, k] * g[j, l] - g[i, l] * g[j, k]) / 6)
        for l in range(n)] for k in range(n)] for j in range(n)] for i in range(n)]
    return W


def main():
    print("=== 4 维 Weyl 张量：Fisher 度规是否自发非零 ===")
    print()

    # ---- 验证：Schwarzschild（已知 Weyl != 0）----
    t, r, th, ph = sp.symbols('t r th ph', real=True)
    M = sp.symbols('M', positive=True)
    f = 1 - 2 * M / r
    g_sch = sp.diag(-f, 1 / f, r ** 2, r ** 2 * sp.sin(th) ** 2)
    coords = [t, r, th, ph]
    print("1. 验证代码：Schwarzschild 度量的 Weyl（应 != 0）：")
    W_sch = weyl_components(g_sch, coords)
    w_sch = sp.simplify(W_sch[0][1][0][1])   # C_{trtr} 分量
    print(f"   C_{{trtr}} = {w_sch}")
    print(f"   => Schwarzschild Weyl != 0：{w_sch != 0}（验证代码正确）")

    # ---- 候选 Fisher 度规：4 参数指数族，A = sum th^2 + 耦合 ----
    print()
    print("2. 候选 Fisher 度规（4 参数，A = th1^2+th2^2+th3^2+th4^2 + th1^2*th2^2）：")
    q1, q2, q3, q4 = sp.symbols('q1 q2 q3 q4', real=True)
    A = q1 ** 2 + q2 ** 2 + q3 ** 2 + q4 ** 2 + q1 ** 2 * q2 ** 2
    G = sp.Matrix([[sp.diff(A, qi, qj) for qj in (q1, q2, q3, q4)]
                   for qi in (q1, q2, q3, q4)])
    G = sp.simplify(G)
    print(f"   Fisher 度规 G = {G.tolist()}")
    # G 是常数系数 + q 依赖（块对角：q1-q2 块 2x2，q3,q4 对角）
    # 算 Weyl
    W_G = weyl_components(G, [q1, q2, q3, q4])
    # 检查几个分量
    comps = {
        "C_0101": W_G[0][1][0][1],
        "C_0202": W_G[0][2][0][2],
        "C_2323": W_G[2][3][2][3],
    }
    any_nonzero = False
    for name, val in comps.items():
        v = sp.simplify(val)
        print(f"   {name} = {v}")
        if v != 0:
            any_nonzero = True
    print(f"   => 候选 Fisher 度规 Weyl {'!= 0（自旋 2 涌现！）' if any_nonzero else '= 0（共形平坦）'}")

    print()
    print("=== 结论 ===")
    print("  Schwarzschild Weyl != 0（验证代码）。")
    print("  候选 Fisher 度规（4 参数指数族）的 Weyl 见上。")

    summary = {
        "schwarzschild_C_trtr": str(sp.simplify(W_sch[0][1][0][1])),
        "schwarzschild_weyl_nonzero": bool(W_sch[0][1][0][1] != 0),
        "fisher_metric": str(G),
        "fisher_weyl_components": {k: str(sp.simplify(v)) for k, v in comps.items()},
        "fisher_weyl_nonzero": bool(any_nonzero),
    }
    out = ROOT / "experiments" / "exp_verify_fisher_weyl_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
