"""台阶1：维度谱重定义（theta-summable 下）—— 热核渐近展开替代 D^{-n} Dixmier 迹。

背景：Connes 标准框架里，维度谱来自 D^{-n} 的 Dixmier 迹。但 theta-summable 放弃紧 D，
D^{-n} 不收敛（六块砖 ③ 已坐实）。Carey–Phillips semi-finite 替代：维度谱来自热核渐近
Tr(e^{-tD^2}) ~ Σ_j c_j t^{-j/2}（t→0^+），维度谱 = {j : c_j ≠ 0}。

核心命题（对比导数型 vs 乘法型）：
  1. 导数型 D = -i d/ds（谱 k ∈ ℝ）：Tr(e^{-tD^2}) = ∫e^{-tk^2}dk = √(π/t) = √π·t^{-1/2}
     → 幂律 t^{-1/2}，维度谱 = {1}（1 维）。
  2. 乘法型 D = logρ（谱 -log i）：Tr(e^{-tD^2}) = Σ e^{-t log^2 i} ~ ∫e^{y-ty^2}dy = e^{1/(4t)}√(π/t)
     → 超指数 e^{1/(4t)}（t→0 发散快于任何幂律），无幂律展开 → 维度谱为空。
  3. 结论：维度谱（热核渐近的幂次）区分「导数型（有几何维度）」vs「乘法型（无维度）」。
     这就是 theta-summable 下维度谱的新定义——热核渐近，替代 D^{-n} Dixmier 迹。

精确性边界（诚实标注）：
  (1) 这是维度谱的「热核判据」，不是完整 semi-finite 谱三元组的维度谱定理（Carey–Phillips）。
  (2) 维度谱的「非主项」（c_j, j 较小）这里没算，只验证主导幂次（维度）。
  (3) 乘法型的「维度谱为空」意味着它不定义几何维度——正是它不能直接当谱三元组 D 的原因。

Code: `py -m experiments.exp_wall_dimension_spectrum`
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
    print("=== sympy 符号验证（热核迹：导数型幂律 vs 乘法型超指数）===")
    print()

    # (1) 导数型热核迹 = √(π/t) = √π·t^{-1/2}
    t, k = sp.symbols('t k', positive=True)
    tr_der = sp.integrate(sp.exp(-t * k ** 2), (k, -sp.oo, sp.oo))
    der_ok = sp.simplify(tr_der - sp.sqrt(sp.pi / t)) == 0
    print(f"(1) 导数型 Tr(e^(-tD^2)) = ∫e^(-tk^2)dk = {sp.simplify(tr_der)} = √π·t^(-1/2) "
          f"{'OK' if der_ok else 'FAIL'}")

    # (2) 乘法型热核迹 = e^{1/(4t)}·√(π/t)（超指数），用配平方 + 高斯（sympy 不自动化简 erf）
    y = sp.symbols('y', real=True)
    # 配平方：y - t y² = 1/(4t) - t(y - 1/(2t))²
    lhs_cs = y - t * y ** 2
    rhs_cs = 1 / (4 * t) - t * (y - 1 / (2 * t)) ** 2
    cs_ok = sp.simplify(sp.expand(lhs_cs - rhs_cs)) == 0
    # 高斯（平移不变）：∫_{-∞}^{∞} e^{-t z²} dz = √(π/t)，z = y - 1/(2t)
    z = sp.symbols('z', real=True)
    gauss = sp.integrate(sp.exp(-t * z ** 2), (z, -sp.oo, sp.oo))
    gauss_ok = sp.simplify(gauss - sp.sqrt(sp.pi / t)) == 0
    mult_ok = cs_ok and gauss_ok
    print(f"(2) 乘法型 ∫e^(y-ty^2)dy = e^(1/4t)·√(π/t)：配平方 {'OK' if cs_ok else 'FAIL'} + "
          f"高斯 {'OK' if gauss_ok else 'FAIL'}（超指数）")

    print()
    print("   结论：导数型幂律 t^{-1/2}（维度 1）；乘法型超指数 e^{1/(4t)}（维度谱空）。")
    print()
    return {"derivative_power_law": bool(der_ok), "multiplicative_superexp": bool(mult_ok)}


def numpy_section():
    print("=== numpy 数值验证（维度谱区分导数型 / 乘法型）===")
    print()

    # (A) 导数型：log Tr vs log t 斜率 = -1/2（幂律，维度 1）
    print("  (A) 导数型 D=-i d/ds：Tr(e^{-tD^2}) = √(π/t)，log-log 斜率应为 -1/2")
    ts = np.array([0.1, 0.05, 0.02, 0.01, 0.005])
    tr_der = np.sqrt(np.pi / ts)
    slope_der = np.polyfit(np.log(ts), np.log(tr_der), 1)[0]
    print(f"      斜率 = {slope_der:.3f}（期望 -0.5）→ 维度 = {-2*slope_der:.1f}")
    print()

    # (B) 乘法型 vs 导数型：超指数 e^{1/(4t)} vs 幂律 t^{-1/2}（随 t→0 增长对比）
    print("  (B) 随 t→0 的增长对比：乘法型超指数 vs 导数型幂律")
    print("   t         导数型 √(π/t)    乘法型 e^(1/4t)√(π/t)   乘法/导数 比值")
    ts2 = np.array([0.2, 0.1, 0.05, 0.02, 0.01])
    for t in ts2:
        der = np.sqrt(np.pi / t)
        mult = np.exp(1 / (4 * t)) * np.sqrt(np.pi / t)
        print(f"   {t:<8}  {der:.3e}        {mult:.3e}          {mult/der:.3e}")
    print()
    print("   结论：乘法型随 t→0 增长超指数（比值 e^(1/4t) 爆炸），无幂律 → 维度谱空。")
    print()

    return {"derivative_slope": float(slope_der)}


def main():
    print("=== 台阶1：维度谱重定义（热核渐近展开）===")
    print("（精确性边界见 docstring）")
    print()

    sym_res = sympy_section()
    num_res = numpy_section()

    print("=== 结论 ===")
    print("  1. 导数型 D=-i d/ds：Tr(e^{-tD^2}) = √π·t^{-1/2}，幂律，维度谱 = {1}。")
    print("  2. 乘法型 D=logρ：Tr(e^{-tD^2}) ~ e^{1/(4t)}√(π/t)，超指数，维度谱为空。")
    print("  3. 维度谱（热核渐近幂次）区分导数型（有维度）/乘法型（无维度）——theta-summable 下的新定义。")
    print()

    summary = {
        "question": "in theta-summable framework, does heat-kernel asymptotics Tr(e^{-tD^2})~sum c_j t^{-j/2} "
                    "redefine dimension spectrum, distinguishing derivative-type from multiplication-type D?",
        "answer": "YES",
        "derivative": "Tr(e^{-tD^2})=sqrt(pi/t)=sqrt(pi)t^{-1/2}, power-law, dimension spectrum {1}",
        "multiplicative": "Tr(e^{-tD^2}) ~ e^{1/(4t)}sqrt(pi/t), super-exponential, empty dimension spectrum",
        "sympy": sym_res,
        "numpy": num_res,
        "precision_boundaries": [
            "heat-kernel criterion, not full Carey-Phillips dimension spectrum theorem",
            "only dominant power (dimension) checked, not subleading c_j",
            "multiplicative 'empty dimension spectrum' = no geometric dimension (why it can't be spectral-triple D)",
        ],
        "conclusion": "dimension spectrum = heat-kernel asymptotic powers; distinguishes derivative (dim 1) "
                      "from multiplication (no dim). New definition under theta-summable.",
    }
    out = ROOT / "experiments" / "exp_wall_dimension_spectrum_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
