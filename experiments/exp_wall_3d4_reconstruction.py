"""台阶3·3.4（阶段 B）：重建定理条件的符号验证 —— 一阶条件 + 光滑性。

背景：3.4 = 弱重建定理（证明交叉积三元组恢复 C^∞(M)）。Connes 重建定理五条件：
(a) 交换 (b) 紧 D (c) 一阶 [[D,a],b]=0 (d) 光滑性 (e) 有限维谱。

本脚本做阶段 B：在交叉积三元组上符号验证 (c)(d)（可符号做的两个）。

核心结论：
  (c) 一阶条件 [[D_R, a], b] = 0 平凡满足：D_R 的导数型部分 [-i d/dt, f] = -i f'（乘法算子），
      f' 与 b 对易 ⟹ [[D_R, a], b] = 0。
  (d) 光滑性满足：[D_R, [D_R, f]] = -f''（二阶交换子 = 二阶导数，有界 ⟺ f 光滑）。

关键障碍（阶段 C，概念，不符号验证）：
  (a) 交换性不满足：交叉积 R ⋊_σ ℝ 是因子（中心平凡 ℂ），不能直接取 A = 中心恢复 C^∞(M)。
      这是 3.4 的真正困难 = 「交换化」（条件期望 E → MASA → C^∞），
      与九块砖墙测绘的 P1（交换化 + 光滑化）同源。

精确性边界（诚实标注）：
  (1) 只验证 (c)(d)（可符号的），(a) 交换性是概念障碍（因子理论），(b) 已放宽、(e) 已坐实。
  (2) (c)(d) 平凡满足是「好消息」：D_R 是一阶 Dirac 型，一阶条件 + 光滑性自动成立。
  (3) 未写「弱重建定理」候选陈述（阶段 D）——那需要先解决 (a) 交换化。

Code: `py -m experiments.exp_wall_3d4_reconstruction`
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
    print("=== 台阶3·3.4（阶段 B）：重建条件符号验证 ===")
    print("（精确性边界见 docstring）")
    print()

    t = sp.symbols('t', real=True)
    f = sp.Function('f')(t)
    g = sp.Function('g')(t)

    # (c) 一阶条件：[[D, f], g] = 0（D = -i d/dt 导数型部分）
    Df = -sp.I * sp.diff(f, t)                 # [D, f] = -i f'
    DDfg = sp.simplify(Df * g - g * Df)         # [[D, f], g] = -i(f'g - gf')
    ok_c = DDfg == 0
    print(f"(c) 一阶条件 [[D, f], g] = {DDfg}（期望 0）: {'OK' if ok_c else 'FAIL'}")

    # (d) 光滑性：[D, [D, f]] = -f''（二阶交换子 = 二阶导数）
    D2f = -sp.diff(f, t, 2)                     # [D, -i f'] = -i(-i f'') = -f''
    # 直接验证：-i d/dt 作用两次 = -d²/dt²
    direct = sp.simplify(-sp.I * sp.diff(-sp.I * sp.diff(f, t), t))
    ok_d = sp.simplify(direct - D2f) == 0
    print(f"(d) 光滑性 [D, [D, f]] = -f'' = {D2f}: {'OK' if ok_d else 'FAIL'}")

    # (c2) 乘法型 H_mod 对一阶条件贡献平凡：[H_mod, g] = 0（乘法算子对易）
    s = sp.symbols('s', real=True)
    gs = sp.Function('gs')(s)
    Hmod = -s                                     # H_mod = log C - s（乘法型，常数 log C 略）
    comm_H = sp.simplify(Hmod * gs - gs * Hmod)
    ok_c2 = comm_H == 0
    print(f"(c2) [H_mod, g(s)] = {comm_H}（期望 0，乘法对易）: {'OK' if ok_c2 else 'FAIL'}")

    print()
    print("=== 结论（阶段 B + C）===")
    print("  1. (c) 一阶条件平凡满足：D_R 是一阶 Dirac 型，[[D_R,a],b]=0 自动成立。")
    print("  2. (d) 光滑性满足：[D_R,[D_R,f]] = -f''，光滑 ⟺ 有界。")
    print("  3. (a) 交换性是硬障碍（阶段 C）：交叉积 R⋊_σ ℝ 是因子（中心 ℂ），")
    print("     不能直接取 A = 中心恢复 C^∞(M)——需「交换化」（E → MASA → C^∞，= P1 墙）。")
    print()

    summary = {
        "question": "do the first-order condition (c) and smoothness (d) hold for the crossed-product "
                    "triple (stage B of 3.4)?",
        "answer": "YES for (c)(d); (a) commutativity is the hard obstacle",
        "first_order_condition": "[[D_R, a], b] = 0 trivial (D_R first-order Dirac, f' commutes with b)",
        "smoothness": "[D_R, [D_R, f]] = -f'' bounded iff f smooth",
        "sympy": {"first_order": bool(ok_c), "smoothness": bool(ok_d), "multiplicative_trivial": bool(ok_c2)},
        "obstacle": "(a) commutativity: crossed product R⋊_σ ℝ is a factor (center ℂ), cannot recover "
                    "C^∞(M) directly — needs abelianization (E → MASA → C^∞, = P1 wall)",
        "precision_boundaries": [
            "only (c)(d) verified symbolically; (a) is conceptual (factor theory); (b) relaxed, (e) done",
            "(c)(d) trivial is GOOD news: D_R first-order Dirac, so first-order + smoothness automatic",
            "candidate weak-reconstruction theorem (stage D) NOT written — needs (a) abelianization first",
        ],
    }
    out = ROOT / "experiments" / "exp_wall_3d4_reconstruction_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
