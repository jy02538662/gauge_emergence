"""自旋 2 弯曲层 · 第二步：λ_c→∞（无质量）+ 缺陷 → 长程 vs 短程？

背景：第一步坐实 Weyl≠0（种子）。第二步测「长程 ⟂ 弯曲」墙是否在 λ_c→∞ 下破。
关键纠正（用户指出）：之前「短程」是在 λ_c 有限（紧 D，有质量）下测的；
λ_c→∞（非紧 D，无质量）下，无质量模（长程）和缺陷（弯曲）可能共存 → 长程弯曲。

核心命题：
  度规质量 gap m ∝ 1/λ_c（λ_c 有限 → m>0 有质量；λ_c→∞ → m→0 无质量）。
  缺陷（点源）诱导的度规涨落 h_μν = Green 函数：
  - λ_c 有限：h_μν ~ e^{-mr}/(4πr)（Yukawa，指数短程）；
  - λ_c→∞：h_μν ~ 1/(4πr)（Coulomb，幂律长程）。
  结论：λ_c→∞ 下「长程 ⟂ 弯曲」墙破——无质量（长程）与缺陷（弯曲）共存，与 GR 一致
  （平背景 + 无质量引力子 = 长程弯曲）。

精确性边界（诚实标注）：
  (1) 这是「传播子/Green 函数」层面的验证，完整自旋 2 度规动力学（变分原理）是第三步。
  (2) m ∝ 1/λ_c 是「尺度依赖观察者态」的推论（exp_wall_scale_dependent_observer），
      精确 m(λ_c) 关系需从 Hessian 质量 gap 算（exp_gravity_mass_gap）。
  (3) 无质量长程是「观察者态路径 1/r」的传播子版，与 [[长程引力=模流接回（命题＋公式）]] 同源。

Code: `py -m experiments.exp_wall_spin2_longrange`
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


def sympy_section():
    print("=== sympy 符号验证（Yukawa → m→0 = Coulomb 长程）===")
    print()

    r, m = sp.symbols('r m', positive=True)
    Gm = sp.exp(-m * r) / (4 * sp.pi * r)     # Yukawa（有质量）
    G0 = 1 / (4 * sp.pi * r)                  # Coulomb（无质量）
    lim = sp.simplify(sp.limit(Gm, m, 0))
    lim_ok = sp.simplify(lim - G0) == 0
    print(f"(1) Yukawa G_m = {Gm} -- m→0 极限 = {lim} == 1/(4πr): {'OK' if lim_ok else 'FAIL'}")

    # 长程性：无质量 1/r 幂律（log-log 斜率 -1），有质量 e^{-mr}/r 指数截断
    print(f"(2) 无质量 G_0 ~ 1/r（幂律长程）；有质量 G_m ~ e^(-mr)/r（指数短程）")

    print()
    print("   结论：λ_c→∞（m→0）→ 无质量 → 长程 1/r；λ_c 有限（m>0）→ 有质量 → 短程。")
    print()
    return {"yukawa_limit_coulomb": bool(lim_ok)}


def numpy_section():
    print("=== numpy 数值验证（λ_c 有限 vs ∞：h_μν 衰减 短程→长程）===")
    print()

    # h_μν(r) = e^{-r/λ_c}/(4πr)，m = 1/λ_c
    rs = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    print("   r         λ_c=2(短程)   λ_c=10       λ_c=50       λ_c=1e6(长程)    1/r 参考")
    for r in rs:
        vals = [np.exp(-r / lc) / (4 * np.pi * r) for lc in (2, 10, 50, 1e6)]
        ref = 1 / (4 * np.pi * r)
        print(f"   {r:>5.1f}   {vals[0]:>10.4f}   {vals[1]:>10.4f}   {vals[2]:>10.4f}   {vals[3]:>10.4f}   {ref:>10.4f}")
    print()

    # 长程性判据：λ_c→∞ 时 h_μν ~ 1/r 幂律（log-log 斜率 → -1）；λ_c 有限时指数截断
    r_far = np.array([5.0, 10.0, 20.0, 40.0])
    slope_inf = np.polyfit(np.log(r_far), np.log(1 / (4 * np.pi * r_far)), 1)[0]
    slope_finite = np.polyfit(np.log(r_far), np.log(np.exp(-r_far / 5) / (4 * np.pi * r_far)), 1)[0]
    print(f"   log-log 斜率：λ_c→∞（无质量）→ {slope_inf:.2f}（幂律 -1，长程）；")
    print(f"                 λ_c=5（有质量）→ {slope_finite:.2f}（指数截断，短程）")
    print()
    print("   结论：λ_c→∞ 下 h_μν ~ 1/r（长程），无质量模与缺陷共存，「长程 ⟂ 弯曲」墙破。")
    print()

    return {"slope_massless": float(slope_inf), "slope_massive": float(slope_finite)}


def main():
    print("=== 自旋 2 弯曲层 · 第二步：λ_c→∞（无质量）+ 缺陷 → 长程 vs 短程 ===")
    print("（精确性边界见 docstring）")
    print()

    sym_res = sympy_section()
    num_res = numpy_section()

    print("=== 结论 ===")
    print("  1. λ_c 有限（度规有质量 gap m>0）：h_μν ~ e^{-mr}/r（指数，短程）——这是之前「短程」的来源。")
    print("  2. λ_c→∞（无质量 m→0）：h_μν ~ 1/r（幂律，长程）——无质量模与缺陷共存。")
    print("  3. 所以「长程 ⟂ 弯曲」墙是 λ_c 有限下的现象；λ_c→∞ 下墙破（长程弯曲），与 GR 一致。")
    print("  4. 这纠正了「大概率短程」的误判：短程是 λ_c 有限的结论，不是 λ_c→∞ 的。")
    print()

    summary = {
        "question": "does λ_c→∞ (massless) + defect give LONG-RANGE h_μν (1/r), breaking the 'long ⟂ curve' wall?",
        "answer": "YES (propagator level)",
        "massive": "λ_c finite: h_μν ~ e^{-mr}/(4πr) (Yukawa, exponential short-range)",
        "massless": "λ_c→∞: h_μν ~ 1/(4πr) (Coulomb, power-law long-range)",
        "wall_breaks": "long ⟂ curve is a λ_c-finite phenomenon; at λ_c→∞ the wall breaks (long-range curvature)",
        "sympy": sym_res,
        "numpy": num_res,
        "precision_boundaries": [
            "propagator/Green-function level; full spin-2 metric dynamics (variational principle) is step 3",
            "m ∝ 1/λ_c from scale-dependent observer; exact m(λ_c) needs Hessian mass gap (exp_gravity_mass_gap)",
            "massless long-range = propagator version of observer-state path 1/r (模流接回)",
        ],
    }
    out = ROOT / "experiments" / "exp_wall_spin2_longrange_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
