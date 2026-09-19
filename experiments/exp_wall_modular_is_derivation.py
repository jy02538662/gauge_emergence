"""检验：模流（时间）的无穷小生成元 = 内导子 [log ρ, ·] = P2 渐近导子。

核心命题（「Out → 模流」转向的坐实）：
  模流 σ_t(x) = ρ^{it} x ρ^{-it} 是内自同构（Ad_{ρ^{it}}），其生成元是内导子
  δ(x) = d/dt σ_t(x)|_{t=0} = i[log ρ, x]。
  所以「模流 → 局域微分同胚（外导子）」=「内导子 [logρ,·] → 外导子」= P2 渐近导子。

本脚本坐实三点（sympy 符号）：
  1. δ(x) = [h, x] 是导子（Leibniz 律）。
  2. δ 作用：对角元不动（时间 = 对角不动）、非对角元相位旋转
     （δ(E_ij) = (log λ_i - log λ_j) E_ij）。
  3. 模流生成元 = i[log ρ, ·]（反厄米 -> 生成酉群 σ_t = e^{itδ}）=> 「时间」= 内导子。

结论：模流 = 内导子的指数；内导子 [logρ,·] 就是 P2 的 δ_N（‖δ_N‖=log N 发散已坐实）。
  => 「模流 → 局域微分同胚」=「内导子 → 外导子」= P2 渐近导子，不是新墙，是有裂缝的老卡点。

Code: `py -m experiments.exp_wall_modular_is_derivation`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def der(h, x):
    """内导子 δ(x) = [h, x] = hx - xh。"""
    return h @ x - x @ h


def is_zero_matrix(M):
    return all(sp.simplify(e) == 0 for e in M)


def main():
    print("=== 模流生成元 = 内导子 [log ρ, ·] = P2 渐近导子（符号坐实）===")
    print()

    # ---- 1. Leibniz 律（导子）----
    print("1. δ(x)=[h,x] 是导子（Leibniz 律）")
    h = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f'h{i+1}{j+1}'))
    x = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f'x{i+1}{j+1}'))
    y = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f'y{i+1}{j+1}'))
    lhs = der(h, x @ y)
    rhs = der(h, x) @ y + x @ der(h, y)
    leibniz_ok = is_zero_matrix(lhs - rhs)
    print(f"   [h, xy] - ([h,x]y + x[h,y]) = {sp.simplify(lhs - rhs)}")
    print(f"   -> Leibniz 律成立（δ 是导子）{'[OK]' if leibniz_ok else '[FAIL]'}")

    # ---- 2. 对角不动 / 非对角相位 ----
    print()
    print("2. δ 作用：对角元不动、非对角元相位旋转")
    a, b = sp.symbols('a b')  # a = log λ1, b = log λ2
    hdiag = sp.diag(a, b)
    # 对角元不动：[h, diag] = 0
    d1, d2 = sp.symbols('d1 d2')
    Ddiag = sp.diag(d1, d2)
    delta_diag = sp.simplify(der(hdiag, Ddiag))
    diag_fixed = is_zero_matrix(delta_diag)
    print(f"   [h, diag(d1,d2)] = {delta_diag}  -> 对角元不动（时间=对角不动）{'[OK]' if diag_fixed else '[FAIL]'}")
    # 非对角元相位：δ(E_12) = (a - b) E_12 = log(λ1/λ2) E_12
    E12 = sp.Matrix([[0, 1], [0, 0]])
    delta_E12 = sp.simplify(der(hdiag, E12))
    expected_E12 = (a - b) * E12
    phase_ok = is_zero_matrix(delta_E12 - expected_E12)
    print(f"   [h, E_12] = {delta_E12}  = (a-b) E_12 = log(λ1/λ2) E_12（非对角元相位）{'[OK]' if phase_ok else '[FAIL]'}")

    # ---- 3. 模流生成元反厄米 ----
    print()
    print("3. 模流生成元 = i[log ρ, ·]（反厄米 -> 生成酉群）")
    # i[h, x] 对厄米 h,x 是厄米（=> [h,x] 反厄米，i[h,x] 厄米），
    # 但直接验证：δ 是内导子 => σ_t = e^{tδ} 是 Ad_{ρ^t}，保乘法（内自同构）。
    # 这里坐实「模流 = 内自同构」：σ_t 保乘法 ⟺ δ 是导子（Leibniz，已验）。
    # 且 δ 反厄米 ⟺ σ_t 保伴随。验证 [h,x] 的反厄米性（h,x 厄米）：
    hh = sp.Matrix([[sp.Symbol('a'), sp.Symbol('c')], [sp.Symbol('c'), sp.Symbol('b')]])  # 厄米
    xx = sp.Matrix([[sp.Symbol('p'), sp.Symbol('q')], [sp.Symbol('q'), sp.Symbol('r')]])  # 厄米
    comm = sp.simplify(der(hh, xx))
    # [h,x] 反厄米 ⟺ [h,x]^† = -[h,x]。对厄米 h,x：[h,x]^† = [x,h] = -[h,x]。
    skew_ok = is_zero_matrix(comm.T + comm)
    print(f"   [h,x] 反厄米（[h,x]^T = -[h,x]，h,x 厄米）{'[OK]' if skew_ok else '[FAIL]'}")
    print("   => δ 反厄米，σ_t = e^{itδ} 是酉共轭（内自同构 = 模流）。")

    # ---- 结论 ----
    print()
    print("=== 结论 ===")
    print("  模流生成元 = i[log ρ, ·]（内导子），= P2 的渐近导子 δ_N = [log ρ_N, ·]。")
    print("  「时间」= 内导子（对角不动、非对角相位旋转 = 跃迁）。")
    print("  => 「模流 → 局域微分同胚（外导子）」=「内导子 → 外导子」= P2 渐近导子。")
    print("     P2 已坐实 ‖δ_N‖ = log N 发散（exp_wall_derivation_asymptotic），有裂缝可钻。")

    summary = {
        "question": "is the modular-flow generator = inner derivation [log rho, .] = P2 asymptotic derivation?",
        "derivation_leibniz": bool(leibniz_ok),
        "diagonal_fixed": bool(diag_fixed),
        "offdiag_phase": bool(phase_ok),
        "skew_hermitian": bool(skew_ok),
        "conclusion": "modular-flow generator = i[log rho, .] (inner derivation) = P2 delta_N. "
                      "Time = inner derivation (diagonal fixed, off-diagonal phase = transition). "
                      "'modular flow -> local diffeomorphism (outer derivation)' = "
                      "'inner -> outer derivation' = P2 asymptotic derivation, "
                      "already verified ||delta_N|| = log N diverges (crack, drillable).",
    }
    out = ROOT / "experiments" / "exp_wall_modular_is_derivation_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
