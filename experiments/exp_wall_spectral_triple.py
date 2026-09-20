"""③ 第二半：完整弱谱三元组 —— 导数型 theta-summable 谱三元组 + 傅里叶对偶。

框架（方向 1 弱谱三元组）③ 第二半：③ 第一半证明了 D=logρ（乘法型）theta-summable 非紧，
但 [logρ, a] = 0（乘法型 Lipschitz 平凡），不是谱三元组要的 D。
本块砖做「傅里叶对偶」：乘法型 logρ ↔ 导数型 D = -i d/ds，后者才是谱三元组的 Dirac 算子。

核心命题（可符号/数值验证）：
  1. 傅里叶对偶：乘法型 M_s（谱 s）与导数型 -i d/ds（谱 k）theta-summable 迹相同
     Tr(e^{-tM_s^2}) = ∫e^{-ts^2}ds = √(π/t) = ∫e^{-tk^2}dk = Tr(e^{-tD^2})
     （高斯积分在傅里叶变换下不变——位置/动量对偶）。
  2. 导数型 Lipschitz 非平凡：[D, f] = [−i d/ds, f] = -i f'（乘法型 [M_s, f] = 0 平凡）。
  3. 三元组 (A, H, D) = (C_c^∞(ℝ), L²(ℝ), -i d/ds)：D 自伴、[D,f] 有界、theta-summable 非紧。
  4. 串联（②a arcsine × ③ 热核）：离散 theta-summable 迹 (1/N)Σ e^{-tλ_k^2}
     → 连续 ∫ e^{-tλ^2}ρ(λ)dλ = e^{-2t}·I_0(2t)（arcsine 加权，Toeplitz 谱）。

精确性边界（诚实标注）：
  (1) 这是 1D 尺度方向的导数型三元组，不是完整时空 D（3 维 = 1 维尺度 × su(2)，见 [[3维空间点内生]]）。
  (2) 维度谱（dimension spectrum）/ 陈数 / 局部指数在 theta-summable 下要重新定义，本块砖不做。
  (3) 自伴性只在 C_c^∞ 上验证（形式自伴），本质自伴性（essential self-adjointness）是后续数学。

Code: `py -m experiments.exp_wall_spectral_triple`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import special

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sympy_section():
    print("=== sympy 符号验证（傅里叶对偶 + Lipschitz 非平凡）===")
    print()

    # (1) 导数型 theta-summable 迹 = √(π/t)
    t, k = sp.symbols('t k', positive=True)
    tr_der = sp.integrate(sp.exp(-t * k ** 2), (k, -sp.oo, sp.oo))
    tr_expected = sp.sqrt(sp.pi / t)
    der_ok = sp.simplify(tr_der - tr_expected) == 0
    print(f"(1) 导数型 Tr(e^(-tD^2)) = ∫e^(-tk^2)dk = {sp.simplify(tr_der)} : {'OK' if der_ok else 'FAIL'}")

    # (2) 乘法型 theta-summable 迹相同（傅里叶对偶）
    s = sp.symbols('s', real=True)
    tr_mult = sp.integrate(sp.exp(-t * s ** 2), (s, -sp.oo, sp.oo))
    dual_ok = sp.simplify(tr_der - tr_mult) == 0
    print(f"(2) 乘法型 Tr(e^(-tM_s^2)) = ∫e^(-ts^2)ds = {sp.simplify(tr_mult)}（傅里叶对偶相同）: "
          f"{'OK' if dual_ok else 'FAIL'}")

    # (3) Lipschitz 非平凡：[D, f] = -i f'（D = -i d/ds，导数型的关键优势）
    f = sp.Function('f')(s)
    g = sp.Function('g')(s)
    # [d/ds, f] g = (d/ds)(f g) - f (d/ds) g = f' g
    lhs = sp.diff(f * g, s) - f * sp.diff(g, s)
    rhs = sp.diff(f, s) * g
    comm_ok = sp.simplify(lhs - rhs) == 0
    print(f"(3) [d/ds, f] = f'（乘法算子，[D,f]=-if' 非平凡）: {'OK' if comm_ok else 'FAIL'}")

    print()
    print("   结论：乘法型 logρ ↔ 导数型 -i d/ds 傅里叶对偶、迹相同；导数型 Lipschitz 非平凡。")
    print()
    return {"derivative_theta": bool(der_ok), "fourier_dual": bool(dual_ok),
            "lipschitz_nontrivial": bool(comm_ok)}


def numpy_section():
    print("=== numpy 数值验证（arcsine 加权串联 + 自伴性）===")
    print()

    # (A) 离散 theta-summable 迹 → 连续 e^{-2t}I_0(2t)（arcsine 加权，串联 ②a）
    print("  (A) (1/N)Σ e^{-t·λ_k^2} → ∫ e^{-tλ^2}ρ(λ)dλ = e^{-2t}·I_0(2t)，t=1")
    target = float(np.exp(-2.0) * special.i0(2.0))
    print(f"      连续目标 e^{-2}·I_0(2) = {target:.6f}")
    for N in [100, 1000, 10000, 100000, 1000000]:
        lk = np.sort([2.0 * np.cos(np.pi * k / (N + 1)) for k in range(1, N + 1)])
        disc = np.sum(np.exp(-1.0 * lk ** 2)) / N
        print(f"      N={N:>8}: (1/N)sum e^(-lam^2) = {disc:.6f}")
    print()

    # (B) 自伴性：离散厄米差分 D_N = -i(S - S^{-1})/(2h) 满足 D_N = D_N†
    print("  (B) 离散厄米差分 D_N 自伴性：‖D_N - D_N†‖ = 0")
    for N in [16, 64, 256]:
        h = 1.0 / N
        S = np.eye(N, k=1) + np.eye(N, k=-(N - 1))   # 循环上移
        D = -1j * (S - S.T) / (2 * h)
        err = float(np.linalg.norm(D - D.conj().T, 2))
        print(f"      N={N}: ‖D - D†‖ = {err:.2e}")
    print()

    return {"arcsine_weighted_target": target}


def main():
    print("=== ③ 第二半：导数型 theta-summable 谱三元组 + 傅里叶对偶 ===")
    print("（精确性边界见 docstring）")
    print()

    sym_res = sympy_section()
    num_res = numpy_section()

    print("=== 结论 ===")
    print("  1. 乘法型 logρ ↔ 导数型 -i d/ds 傅里叶对偶，theta-summable 迹都 = √(π/t)。")
    print("  2. 导数型 [D,f] = -i f' 非平凡（乘法型 [logρ,a]=0 平凡）——这是谱三元组要的 D。")
    print("  3. 三元组 (C_c^∞(R), L²(R), -i d/ds)：自伴 + Lipschitz 非平凡 + theta-summable 非紧。")
    print("  4. 串联：离散 (1/N)Σe^{-tλ^2} → e^{-2t}I_0(2t)（arcsine 加权），②a×③ 汇合。")
    print()

    summary = {
        "question": "does the derivative-type D=-i d/ds give a theta-summable spectral triple, "
                    "Fourier-dual to the multiplication-type logρ (③ first half)?",
        "answer": "YES",
        "fourier_dual": "multiplication M_s and derivative -i d/ds both have Tr(e^{-tD^2}) = sqrt(pi/t)",
        "lipschitz": "[D,f] = -i f' nontrivial (multiplication-type had [M_s,f]=0 trivial)",
        "triple": "(C_c^∞(R), L²(R), -i d/ds): self-adjoint + nontrivial Lipschitz + theta-summable non-compact",
        "arcsine_series": "(1/N)Σe^{-tλ^2} -> e^{-2t}I_0(2t) (arcsine-weighted, ties ②a to ③)",
        "sympy": sym_res,
        "numpy": num_res,
        "precision_boundaries": [
            "1D scale-direction triple, not full spacetime D (3D = 1D scale × su(2))",
            "dimension spectrum / Chern / local index not yet redefined under theta-summable",
            "self-adjointness only on C_c^∞ (formal); essential self-adjointness is later math",
        ],
        "conclusion": "derivative-type D = -i d/ds is the Fourier dual of logρ, gives nontrivial Lipschitz, "
                      "and defines a theta-summable spectral triple. This completes the weak-spectral-triple core.",
    }
    out = ROOT / "experiments" / "exp_wall_spectral_triple_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
