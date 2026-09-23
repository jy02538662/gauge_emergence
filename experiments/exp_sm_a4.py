"""B2: 从 D_{R(x)M_3} 热核提取 a_4 的规范场 F^2 项（Yang-Mills 动力学来源）。

v8 第二层 B2。含颜色规范场的 Dirac 算子，热核 a_4 会多出规范场 F^2 项。
本脚本符号验证 a_4 的 F^2 系数（不手推 gamma 迹）。

含规范场的 Dirac（Vassilevich Eq 3.27/3.28，无轴矢量 A5=0）：
  E   = -R/4 · I + (1/4)[γ^μ,γ^ν]F_{μν}
  Ω_{μν} = F_{μν} - (1/4)γ^σγ^ρ R_{σρμν}
其中 F_{μν} 是颜色空间的规范曲率（3×3，作用在颜色指标），[γ,γ] 是旋量矩阵。

a_4（Vassilevich Eq 4.28）的 F^2 项来自 180E^2 和 30Ω^2：
  tr(E^2)_{F^2} = (1/16) tr_spinor([γ,γ][γ,γ]) tr_color(FF) = -2 tr(F^2)
  tr(Ω^2)_{F^2} = tr_spinor(I) tr_color(FF) = 4 tr(F^2)
  （tr(F^2) = tr_color(F_{μν}F^{μν})）

a_4 的 F^2 项 = (4π)^{-2}(1/360)∫ [180·(-2) + 30·4] tr(F^2) = -(1/24π^2)∫ tr(F^2)

符号验证：用 sympy 显式 gamma 矩阵，符号算 tr(E^2) 和 tr(Ω^2) 的 F^2 项，
不手推 gamma 迹。最后与 Vassilevich 标准（Dirac 基本表示 F^2 系数）对比。

Code: `py -m experiments.exp_sm_a4`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import sympy as sp

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def euclidean_gamma():
    """4x4 欧氏 Dirac 矩阵。"""
    I = sp.eye(2)
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])
    return [
        sp.kronecker_product(sx, sx),
        sp.kronecker_product(sx, sy),
        sp.kronecker_product(sx, sz),
        sp.kronecker_product(sy, I),
    ]


def commutator_trace_identity(gammas):
    """符号验证 tr([γ^μ,γ^ν][γ^α,γ^β]) = 16(-δ^{μα}δ^{νβ}+δ^{μβ}δ^{να})。"""
    ok = True
    for mu in range(4):
        for nu in range(4):
            for al in range(4):
                for be in range(4):
                    c1 = gammas[mu] * gammas[nu] - gammas[nu] * gammas[mu]
                    c2 = gammas[al] * gammas[be] - gammas[be] * gammas[al]
                    lhs = sp.trace(c1 * c2)
                    rhs = 16 * (-(1 if mu == al else 0) * (1 if nu == be else 0)
                                + (1 if mu == be else 0) * (1 if nu == al else 0))
                    if sp.simplify(lhs - rhs) != 0:
                        ok = False
    return ok


def build_E(gammas, F, R):
    """E = -R/4·I + (1/4)Σ[γ^μ,γ^ν]F_{μν}。F 是 4×4 反对称符号矩阵。"""
    I4 = sp.eye(4)
    E = -sp.Rational(1, 4) * R * I4
    for mu in range(4):
        for nu in range(4):
            c = gammas[mu] * gammas[nu] - gammas[nu] * gammas[mu]
            E += sp.Rational(1, 4) * c * F[mu, nu]
    return E


def build_Omega(gammas, F, R, s4_curvature=True):
    """Ω_{μν} = F_{μν}·I - (1/4)Σγ^σγ^ρ R_{σρμν}。R 是曲率（S^4 常数曲率用于 R 项）。"""
    I4 = sp.eye(4)
    Omega = {}
    for mu in range(4):
        for nu in range(4):
            O = F[mu, nu] * I4
            for sigma in range(4):
                for rho in range(4):
                    c = gammas[sigma] * gammas[rho]  # 注意：Vassilevich 用 γ^σγ^ρ，非反对易子
                    Rval = s4_curvature_val(sigma, rho, mu, nu, R) if s4_curvature else 0
                    O += -sp.Rational(1, 4) * c * Rval
            Omega[(mu, nu)] = O
    return Omega


def s4_curvature_val(sigma, rho, mu, nu, R):
    """S^4 单位半径曲率 R_{σρμν} = R/12·(δ_{σμ}δ_{ρν}-δ_{σν}δ_{ρμ})。"""
    return sp.Rational(1, 12) * R * (
        (1 if sigma == mu else 0) * (1 if rho == nu else 0)
        - (1 if sigma == nu else 0) * (1 if rho == mu else 0)
    )


def main():
    print("=== B2: a_4 的规范场 F^2 项（Yang-Mills 动力学来源）===")
    print()

    gammas = euclidean_gamma()
    # F_{μν} 反对称符号（6 个独立分量）
    F = sp.MutableDenseNDimArray.zeros(4, 4)
    R = sp.symbols("R", real=True)
    f01, f02, f03, f12, f13, f23 = sp.symbols("f01 f02 f03 f12 f13 f23", real=True)
    F[0, 1] = f01; F[1, 0] = -f01
    F[0, 2] = f02; F[2, 0] = -f02
    F[0, 3] = f03; F[3, 0] = -f03
    F[1, 2] = f12; F[2, 1] = -f12
    F[1, 3] = f13; F[3, 1] = -f13
    F[2, 3] = f23; F[3, 2] = -f23

    # ---- 1. 符号验证 γ 反对易子迹恒等式（复用第一层，确认 F^2 项的依据）----
    print("1. 符号验证 tr([γ^μ,γ^ν][γ^α,γ^β]) = 16(-δ^{μα}δ^{νβ}+δ^{μβ}δ^{να})")
    id_ok = commutator_trace_identity(gammas)
    print(f"   {'OK' if id_ok else 'FAIL'}（256 组指标）")

    # ---- 2. 符号算 tr(E^2) 的 F^2 项 ----
    print()
    print("2. tr(E^2) 的 F^2 项（E = -R/4·I + (1/4)[γ,γ]F）")
    E = build_E(gammas, F, R)
    E2 = sp.simplify(E * E)
    trE2 = sp.simplify(sp.trace(E2))
    # 提取 F^2 项（f 符号的二次项，不含 R）
    trE2_F2 = sp.expand(trE2)
    # F^2 = 2(f01^2+f02^2+f03^2+f12^2+f13^2+f23^2)（tr_color(F_{μν}F^{μν}) 的归一）
    # 这里用符号验证 trE2 的 F^2 系数：把 trE2 按 f^2 项展开，看系数
    F2_expr = f01**2 + f02**2 + f03**2 + f12**2 + f13**2 + f23**2
    # tr(F_{μν}F^{μν}) = 2·(f01²+f02²+f03²-f12²-f13²-f23²)... 实际颜色迹归一不同
    # 这里直接看 trE2 的 f 二次项系数
    coeff = trE2_F2.coeff(f01, 2)
    print(f"   tr(E^2) 的 f01^2 项系数 = {sp.simplify(coeff)}")
    # 期望：tr(E^2)_{F^2} = -2 tr(F^2)，其中 tr(F^2) = 2(f01²+...)-... 需明确归一
    # 直接验证：tr(E^2) 的 F^2 部分 = -(1/2) tr_spinor([γ,γ][γ,γ])/16 展开
    print("   （F^2 项系数由 tr([γ,γ][γ,γ]) 恒等式直接给出，见下组装）")

    # ---- 3. 符号算 tr(Ω^2) 的 F^2 项 ----
    print()
    print("3. tr(Ω^2) 的 F^2 项（Ω = F·I - (1/4)γγR）")
    Omega = build_Omega(gammas, F, R)
    trOmega2 = 0
    for mu in range(4):
        for nu in range(4):
            trOmega2 += sp.trace(Omega[(mu, nu)] * Omega[(mu, nu)])
    trOmega2 = sp.simplify(trOmega2)
    coeff_O = sp.expand(trOmega2).coeff(f01, 2)
    print(f"   tr(Ω^2) 的 f01^2 项系数 = {sp.simplify(coeff_O)}")

    # ---- 4. 组装 a_4 的 F^2 项 ----
    print()
    print("4. a_4 的 F^2 项组装（Vassilevich Eq 4.28: 180E^2 + 30Ω^2）")
    # tr(F^2) = tr_color(F_{μν}F^{μν})，作为独立符号
    # 手推（用符号验证过的恒等式）：tr(E^2)_{F^2} = -2 tr(F^2)，tr(Ω^2)_{F^2} = 4 tr(F^2)
    # 程序验证这个：用显式 f 符号算 tr(E^2) 和 tr(Ω^2) 的 F^2 部分，对比 -2·F2norm 和 4·F2norm
    # F^2 归一：tr_color(F_{μν}F^{μν}) = Σ_{μν} F_{μν}F_{μν}（颜色迹，这里 F 是标量符号）
    F2norm = 2 * (f01**2 + f02**2 + f03**2 + f12**2 + f13**2 + f23**2)  # Σ_{μν}F_{μν}F^{μν} = 2Σ_{i<j}f_{ij}^2
    # tr(E^2) 的 F^2 部分（纯 f 二次项）
    trE2_f2 = sp.expand(trE2).subs({R: 0})
    trO2_f2 = sp.expand(trOmega2).subs({R: 0})
    cE = sp.simplify(trE2_f2 / F2norm)  # 应该 = -2
    cO = sp.simplify(trO2_f2 / F2norm)  # 应该 = 4
    print(f"   tr(E^2)_F2 / tr(F^2) = {cE} (expect -2)")
    print(f"   tr(Omega^2)_F2 / tr(F^2) = {cO} (expect 4)")

    a4_F2_coeff = sp.Rational(1, 360) * (180 * cE + 30 * cO)
    a4_F2_coeff = sp.simplify(a4_F2_coeff)
    print(f"   a_4 F^2 coefficient = (1/360)(180*{cE} + 30*{cO}) = {a4_F2_coeff} (rel to (4pi)^-2 integral tr(F^2))")
    print(f"   i.e. a_4 contains {a4_F2_coeff}*(4pi)^-2 integral tr(F^2) = {sp.simplify(a4_F2_coeff/16/sp.pi**2)} integral tr(F^2)/pi^2")

    # ---- 结论 ----
    print()
    print("=== 结论 ===")
    print("  a_4 的 F^2 项（规范场/Yang-Mills 动力学来源）符号验证得到。")
    print("  这是谱作用量 Tr f(D²/Λ²) 在 a_4 阶给出 Yang-Mills 规范动力学的机制。")

    summary = {
        "question": "extract the gauge-field F^2 term of a_4 (Yang-Mills dynamics source) from the color-coupled Dirac",
        "gamma_identity_ok": bool(id_ok),
        "trE2_F2_ratio": str(cE),
        "trOmega2_F2_ratio": str(cO),
        "a4_F2_coefficient": str(a4_F2_coeff),
        "conclusion": "a_4 contains -(1/24π²)∫tr(F²) gauge term (Yang-Mills dynamics), symbolically verified "
                      "via explicit gamma matrices (tr E²=-2trF², tr Ω²=4trF²).",
    }
    out = ROOT / "experiments" / "exp_sm_a4_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
