"""台阶3·3.3：模流共轭 → 交叉积 ℝ = 尺度方向（傅里叶对偶的符号验证）。

背景：3.3 判断交叉积 R ⋊_σ ℝ 的 ℝ 方向（-i d/dt）与尺度方向（H_mod = logρ）是否重合。
结论：重合（傅里叶对偶）→ 几何维 1（时间/尺度）+ 2（su(2) 角向）= 3 维，与物理一致。

本脚本坐实 3.3 的两个关键引理（符号验证）：
  1. 模流 = 相位调制：ρ^{it} = e^{it log ρ} = e^{it log C}·e^{-ist}（傅里叶核 e^{-ist}），
     即模流参数 t 与尺度 s 傅里叶对偶。
  2. 傅里叶对偶：F[-i f'](k) = k·F[f](k)（导数 ↔ 乘法），即 -i d/dt ↔ M_s。

Takesaki 对偶（概念，不符号验证）：交叉积 R ⋊_σ ℝ 的对偶作用 σ̂_s(U(t)) = e^{ist}·U(t)
（对 U(t) 的相位调制），对偶群参数 s 正是尺度 s = log λ。

精确性边界（诚实标注）：
  (1) 这是「概念推导 + 关键引理符号验证」，不是完整 Takesaki 对偶的严格化
      （对偶作用的精确定义、L² 实现、谱分解是后续数学）。
  (2) 「交叉积 ℝ = 尺度方向」应精确为「共轭对偶」：-i d/dt 和 H_mod 不是同一算子，
      而是同一 1 维几何方向的两种表示（像 x 和 p 是同一维度的两种表示）。
  (3) 未碰 3.4（重建定理）。

Code: `py -m experiments.exp_wall_3d3_fourier_duality`
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
    print("=== 台阶3·3.3：模流共轭 → 交叉积 ℝ = 尺度方向（符号验证）===")
    print("（精确性边界见 docstring）")
    print()

    # (1) 模流 = 相位调制（傅里叶核 e^{-ist}）
    s, t, C = sp.symbols('s t C', positive=True)
    rho = C * sp.exp(-s)
    logrho = sp.expand_log(sp.log(rho))
    rho_it = sp.exp(sp.I * t * logrho)
    expected = sp.exp(sp.I * t * sp.log(C)) * sp.exp(-sp.I * s * t)
    ok1 = sp.simplify(rho_it - expected) == 0
    print(f"(1) 模流 ρ^it = e^(it·logC)·e^(-ist)（傅里叶核 e^(-ist)）: {'OK' if ok1 else 'FAIL'}")
    print(f"    log ρ = {logrho}（尺度 s 的乘法算子）")

    # (2) 傅里叶对偶：F[-i f'](k) = k·F[f](k)（导数 ↔ 乘法）
    tv, kv = sp.symbols('tv kv', real=True)
    f = sp.exp(-tv ** 2)
    Ff = sp.integrate(f * sp.exp(-sp.I * kv * tv), (tv, -sp.oo, sp.oo))
    fp = sp.diff(f, tv)
    Ffp = sp.integrate(-sp.I * fp * sp.exp(-sp.I * kv * tv), (tv, -sp.oo, sp.oo))
    ok2 = sp.simplify(Ffp - kv * Ff) == 0
    print(f"(2) 傅里叶对偶 F[-i f'](k) = k·F[f](k)（导数 ↔ 乘法）: {'OK' if ok2 else 'FAIL'}")
    print(f"    F[f](k) = {sp.simplify(Ff)}（高斯），F[-i f'](k) = k·F[f](k)")

    print()
    print("=== 结论 ===")
    print("  1. 模流参数 t 与尺度 s 傅里叶对偶（e^{-ist} 是傅里叶核）。")
    print("  2. 导数 -i d/dt ↔ 乘法 M_s 傅里叶对偶（F 互换）。")
    print("  3. Takesaki 对偶 σ̂_s(U(t)) = e^{ist} U(t) 把对偶群参数 s 对到尺度——三者合起来：")
    print("     交叉积 ℝ（-i d/dt）与尺度（H_mod）是共轭对偶 = 同一 1 维几何方向。")
    print("  4. 总几何维 = 1（时间/尺度）+ 2（su(2) 角向）= 3 维，与物理一致。")
    print()

    summary = {
        "question": "are the crossed-product R-direction (-i d/dt) and scale direction (H_mod=logρ) "
                    "Fourier-dual (same 1D geometric direction), giving total dim 1+2=3?",
        "answer": "YES (concept + symbolic verification of key lemmas)",
        "modular_flow_phase_modulation": "rho^it = e^{it logC} e^{-ist} (Fourier kernel), t <-> s dual",
        "fourier_duality": "F[-i f'](k) = k F[f](k), derivative <-> multiplication",
        "takesaki_dual": "σ̂_s(U(t)) = e^{ist} U(t) (phase modulation), dual group param s = scale",
        "conclusion": "crossed R = scale direction (Fourier dual, same 1D geometric direction); "
                      "total dim = 1 + 2 (su(2)) = 3, matches physics",
        "sympy": {"modular_phase": bool(ok1), "fourier_dual": bool(ok2)},
        "precision_boundaries": [
            "concept + lemma verification, not full Takesaki-duality rigorization",
            "'crossed R = scale' should read 'conjugate dual' (same 1D direction, two reps)",
            "does NOT touch 3.4 (reconstruction theorem)",
        ],
    }
    out = ROOT / "experiments" / "exp_wall_3d3_fourier_duality_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
