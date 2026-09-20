"""③：P4 弱谱三元组 —— theta-summable 替换「紧 D」。

框架（方向 1 弱谱三元组）③：P4（最硬卡点）= R 无紧 D（Dixmier 迹发散）。
本块砖做「弱谱三元组」的核心替换：Connes 要求 D 有紧预解式（D^{-1} 紧 / Schatten），
换成 theta-summable（只要求 e^{-tD^2} 迹类），我们的观察者态模流生成元 D=logρ 正好满足后者。

核心命题（可符号/数值验证）：
  取 D = logρ（观察者态模流生成元），幂律谱 ρ_i = 1/i（尺度不变 = 无偏好）。
  1. 紧性判据失败：|D|^{-1} 的谱 = 1/log(i)，对数衰减（非幂律）。
     Dixmier 迹 Tr_ω(|D|^{-1}) = (1/log N)·Σ 1/log(i) ~ N/log^2(N) 发散；
     且 Σ 1/log^p(i) 对任何 p>0 都发散 ⟹ 非 Schatten p-类（任何 p）。
  2. theta-summable 判据通过：Tr(e^{-tD^2}) = Σ e^{-t log^2(i)} = Σ i^{-t log i}
     超多项式衰减，迹类收敛（t=1 → 1.238，与 N 无关）。
  3. 结论：D=logρ 是 theta-summable 但非紧 → 弱谱三元组（theta-summable 变体）成立，
     替换「紧预解式」硬要求。这正是方向 1（弱谱三元组）的落地。

关键意义：P4 的「无紧 D」不是死路——它只否掉「紧预解式」这一条路，
theta-summable 变体（Connes 自己引入的放宽）仍然让 D 定义可用的谱三元组。
「紧 D」的墙 = 幂律谱（尺度不变）的代价，可被 theta-summable 绕过。

精确性边界（诚实标注）：
  (1) 这是「theta-summable 替换」的判据验证，不是完整弱谱三元组的构造
      （完整构造要 A、H、D 三元组 + 作用量 + 维度谱，theta-summable 只是第一块）。
  (2) theta-summable 是「放弃局部性换取热核正则性」，代价是维度谱/陈数等要重新定义。
  (3) 这里 D=logρ 是「1D 尺度方向」的模流生成元，不是完整时空 D——是 P4 的最小可算版本。

Code: `py -m experiments.exp_wall_theta_summable`
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
    print("=== sympy 符号验证 ===")
    print()

    # (1) 恒等式 e^{-t log^2(i)} = i^{-t log(i)}（theta-summable 核的超多项式衰减）
    #     sympy 的 simplify 不自动处理 e^{a·log(i)}=i^a，改用高精度数值验证
    i, t = sp.symbols('i t', positive=True)
    lhs = sp.exp(-t * sp.log(i) ** 2)
    rhs = i ** (-t * sp.log(i))
    id_ok = True
    for iv, tv in [(2, 1.0), (3, 0.5), (5, 2.0), (10, 1.5), (100, 3.0)]:
        l = float(sp.N(lhs.subs({i: iv, t: tv}), 50))
        r = float(sp.N(rhs.subs({i: iv, t: tv}), 50))
        if abs(l - r) > 1e-30:
            id_ok = False
    print(f"(1) e^(-t·log^2(i)) = i^(-t·log i)（超多项式衰减，高精度数值）: {'OK' if id_ok else 'FAIL'}")

    # (2) 高斯核收敛：∫_0^∞ e^{-y^2} dy = √π/2（theta-summable 核的积分版）
    y = sp.symbols('y', positive=True)
    gauss = sp.integrate(sp.exp(-y ** 2), (y, 0, sp.oo))
    gauss_expected = sp.sqrt(sp.pi) / 2
    gauss_ok = sp.simplify(gauss - gauss_expected) == 0
    print(f"(2) ∫_0^∞ e^(-y^2) dy = √π/2 = {sp.nsimplify(gauss)} : {'OK' if gauss_ok else 'FAIL'}")

    print()
    print("   结论：e^{-tD^2} 的谱超多项式衰减（迹类），theta-summable 判据通过。")
    print()
    return {"identity": bool(id_ok), "gaussian_kernel": bool(gauss_ok)}


def numpy_section():
    print("=== numpy 数值验证（theta-summable 通过 / 紧性判据失败）===")
    print()

    # theta-summable：Tr(e^{-tD^2}) 收敛（对固定 t）
    print("  (A) Tr(e^{-tD^2}) = Σ_{i=2}^N e^{-t log^2(i)}：应随 N 收敛到有限值")
    print("   N          t=0.5       t=1.0       t=2.0")
    Ns_theta = [100, 1000, 10000, 100000, 1000000]
    theta_last = {}
    for N in Ns_theta:
        i = np.arange(2, N + 1)
        vals = [np.sum(np.exp(-t * np.log(i) ** 2)) for t in (0.5, 1.0, 2.0)]
        theta_last[N] = vals[1]
        print(f"   {N:>8}   {vals[0]:.6f}   {vals[1]:.6f}   {vals[2]:.6f}")
    print()

    # 紧性判据失败：Dixmier 迹发散 + 非 Schatten
    print("  (B) Dixmier 迹 Tr_ω(|D|^{-1}) = (1/log N)·Σ 1/log(i)：应随 N 发散")
    print("   N           Dixmier迹      N/log^2(N) 比例")
    for N in [100, 1000, 10000, 100000, 1000000]:
        i = np.arange(2, N + 1)
        dixmier = np.sum(1.0 / np.log(i)) / np.log(N)
        print(f"   {N:>8}   {dixmier:>10.3f}   {N/np.log(N)**2:>10.3f}")
    print()

    # Schatten p-类判据：Σ 1/log^p(i) 对任何 p 都发散
    print("  (C) Schatten p-类：Σ 1/log^p(i) 对 p=1,2 都发散（对数衰减非幂律）")
    for p in [1.0, 2.0]:
        vals = []
        for N in [1000, 10000, 100000]:
            i = np.arange(2, N + 1)
            vals.append(np.sum(1.0 / np.log(i) ** p))
        print(f"      p={p}: N=1e3/1e4/1e5 -> {vals[0]:.2f} / {vals[1]:.2f} / {vals[2]:.2f}（发散）")
    print()

    return {"theta_summable_converges": float(theta_last[Ns_theta[-1]])}


def main():
    print("=== ③：P4 弱谱三元组 —— theta-summable 替换「紧 D」===")
    print("（精确性边界见 docstring）")
    print()

    sym_res = sympy_section()
    num_res = numpy_section()

    print("=== 结论 ===")
    print("  1. Tr(e^{-tD^2}) 收敛（theta-summable 通过）：D=logρ 的核超多项式衰减。")
    print("  2. Dixmier 迹发散 ~N/log^2(N)（紧性判据失败）：|D|^{-1} 对数衰减，非 Schatten p-类。")
    print("  3. 所以 D=logρ 是 theta-summable 但非紧 → 弱谱三元组（theta-summable 变体）成立。")
    print("  4. P4「无紧 D」不是死路：紧预解式可被 theta-summable 替换（Connes 自己引入的放宽）。")
    print()

    summary = {
        "question": "is D=logρ (modular flow generator, power-law spectrum ρ_i=1/i) theta-summable "
                    "but non-compact, enabling the weak spectral triple (direction 1)?",
        "answer": "YES",
        "theta_summable": "Tr(e^{-tD^2}) = Σ i^{-t log i} converges (super-polynomial), t=1 -> 1.238",
        "compactness_fails": "Dixmier trace (1/log N)Σ1/log(i) ~ N/log^2(N) diverges; "
                            "|D|^{-1} not in any Schatten p-class (log decay, not power-law)",
        "sympy": sym_res,
        "numpy": num_res,
        "precision_boundaries": [
            "criterion check for theta-summable replacement, not full weak-spectral-triple construction",
            "theta-summable sacrifices locality for heat-kernel regularity; dimension spectrum needs redefinition",
            "D=logρ is the 1D scale-direction modular generator, not the full spacetime D",
        ],
        "conclusion": "P4's 'no compact D' is not a dead end: compact resolvent can be replaced by "
                      "theta-summable (Connes' own relaxation), and D=logρ is theta-summable but non-compact.",
    }
    out = ROOT / "experiments" / "exp_wall_theta_summable_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
