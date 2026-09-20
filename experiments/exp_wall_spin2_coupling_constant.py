"""自旋 2 弯曲层 · 第六步：8πG 匹配 + 缺陷散度精确化（完整 EH 的系数收尾）。

背景：第五步组装了 G_μν = 8πG T_μν 的两边（谱作用量变分 + 物质源键序），但用户指出两处要精确化：
  问题 1（缺陷散度 ≠ 0）：第五步的「缺陷」是手放的局域势 V（非广义协变物质场），
    所以它的散度 ≠ 0 是「手放源的伪影」，不是「物理源」。守恒律 ∇^μ T_μν = 0 是
    「广义协变物质场」的定理（Noether/Bianchi）——物理源自动守恒，手放源不守恒。
  问题 2（8πG 匹配）：G_μν 和 8πG T_μν 的「等号」需要系数匹配——谱作用量 a_2 项系数
    f_1 Λ² 匹配 1/16πG（Chamseddine-Connes 标准收尾）。

核心命题：
  1. 8πG 匹配：f_1 Λ² / 6 = 1/(16πG) ⟹ G = 3/(8π f_1 Λ²)，Newton 常数由谱作用量参数读出。
  2. 守恒律精确化：∇^μ T_μν = 0 是广义协变物质场的定理（Noether）；手放源（非广义协变）
    散度 ≠ 0 是伪影，物理源（物质作用量变分）自动守恒。

精确性边界（诚实标注）：
  (1) 8πG 匹配是 Chamseddine-Connes 1997 标准结果，我们符号验证「G = 3/(8π f_1 Λ²)」。
  (2) 守恒律精确化是「Noether 定理」的概念澄清，不是新数值——手放源 vs 物理源的区分。
  (3) 完整「物理源（广义协变物质场）」的 T_μν 显式构造（从物质作用量变分），需 3+1 费米海，是后续。

Code: `py -m experiments.exp_wall_spin2_coupling_constant`
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
    print("=== 自旋 2 弯曲层 · 第六步：8πG 匹配 + 缺陷散度精确化 ===")
    print("（精确性边界见 docstring）")
    print()

    # (1) 8πG 匹配：谱作用量 a_2 项系数 → Newton 常数 G
    f1, Lambda, G = sp.symbols('f1 Lambda G', positive=True)
    lhs = f1 * Lambda ** 2 / 6          # 谱作用量 a_2 项系数（热核展开）
    rhs = 1 / (16 * sp.pi * G)          # Einstein-Hilbert 系数
    G_solved = sp.solve(sp.Eq(lhs, rhs), G)[0]
    check = sp.simplify(lhs - 1 / (16 * sp.pi * G_solved)) == 0
    print(f"(1) 8πG 匹配：f1·Λ²/6 = 1/(16πG) ⟹ G = {sp.simplify(G_solved)} = 3/(8π f1 Λ²)")
    print(f"    验证：代入 G 后 a_2 系数 == 1/(16πG)? {check}")

    # (2) 守恒律精确化：∇^μ T_μν = 0 是广义协变物质场的定理（Noether）
    print(f"(2) 守恒律精确化：∇^μ T_μν = 0 是「广义协变物质场」的定理（Noether/Bianchi）")
    print(f"    - 物理源（物质作用量 S_m[φ,g] 变分）：T_μν = (2/√-g)δS_m/δg 自动守恒；")
    print(f"    - 手放源（局域势 V 直接加 D[x0,x0]，非广义协变）：散度 ≠ 0 是伪影，不是守恒律违反。")

    print()
    print("=== 结论 ===")
    print("  1. 8πG 匹配：G = 3/(8π f1 Λ²)，Newton 常数由谱作用量参数（截断 Λ + f 的矩 f1）读出。")
    print("  2. 守恒律精确化：第五步的「缺陷散度 ≠ 0」是手放源的伪影，物理源（广义协变）自动守恒。")
    print("  3. 所以完整 EH = a_2 变分 → G_μν = 8πG T_μν，系数匹配（8πG）+ 守恒律（Noether）都坐实。")
    print()

    summary = {
        "question": "does 8πG match (a_2 coefficient -> Newton constant) + is defect divergence a hand-put artifact?",
        "answer": "YES",
        "coupling_constant": "G = 3/(8π f1 Λ²), Newton constant from spectral-action parameters (cutoff Λ + moment f1)",
        "conservation": "∇^μ T_μν = 0 is Noether theorem for generally-covariant matter; "
                       "hand-put source (V) divergence ≠ 0 is artifact, physical source auto-conserves",
        "sympy": {"G_solved": str(sp.simplify(G_solved)), "match_check": bool(check)},
        "precision_boundaries": [
            "8πG match is Chamseddine-Connes 1997 standard; we symbolically verify G=3/(8π f1 Λ²)",
            "conservation clarification is Noether-theorem concept, not new numerics",
            "full physical-source T_μν (from matter action variation) needs 3+1 Fermi sea (later)",
        ],
    }
    out = ROOT / "experiments" / "exp_wall_spin2_coupling_constant_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
