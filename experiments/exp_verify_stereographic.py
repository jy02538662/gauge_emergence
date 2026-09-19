"""符号推：S^3 的 Bures 度量在球极投影下变成什么度规。

S^3 圆度量 -> 球极投影 -> R^3 共形平坦度量。
验证：共形平坦 + 标量曲率非零（弯曲）+ 3 维 Weyl 恒 0（引力波问题）。

Code: `py -m experiments.exp_verify_stereographic`
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
    print("=== 符号推：球极投影下的 Bures 度量 ===")
    print()

    # ---- 球极投影：S^3 单位半径 -> R^3 ----
    # S^3 圆度量 ds^2 = dx0^2 + dx1^2 + dx2^2 + dx3^2（单位半径）
    # 球极投影（去北极）：y_i = x_i/(1-x0)
    # 度量 -> ds^2 = 4/(1+r^2)^2 (dr^2 + r^2 dOmega^2)，r = |y|
    r, theta = sp.symbols('r theta', real=True, positive=True)
    Omega = 2 / (1 + r ** 2)          # 共形因子
    print("1. 球极投影度量 = 共形平坦：")
    print(f"   S^3 圆度量 -> ds^2 = (4/(1+r^2)^2)(dr^2 + r^2 dOmega^2)")
    print(f"   共形因子 Omega = 2/(1+r^2) = {Omega}")
    print(f"   => g_ij = Omega^2 * delta_ij（共形平坦，非欧氏）")

    # ---- 标量曲率（共形变换公式，n=3，欧氏 R=0）----
    print()
    print("2. 标量曲率（共形平坦的非零弯曲）：")
    ln_Omega = sp.log(2) - sp.log(1 + r ** 2)
    d1 = sp.diff(ln_Omega, r)                    # d ln Omega / dr
    d2 = sp.diff(d1, r)                          # d^2 ln Omega / dr^2
    Delta_ln = d2 + (2 / r) * d1                 # 球对称拉普拉斯（3 维）
    n = 3
    # 共形变换 g' = Omega^2 g：R' = Omega^-2 [R - 2(n-1) Delta ln Omega - (n-1)(n-2) |grad ln Omega|^2]
    # 欧氏 R=0；|grad ln Omega|^2 = (d1)^2（欧氏梯度）
    R = sp.simplify((-2 * (n - 1) * Delta_ln - (n - 1) * (n - 2) * (d1) ** 2) / Omega ** 2)
    print(f"   标量曲率 R = {R}")
    print(f"   => R 是常数（= 6，S^3 常曲率 K=1），不是逐点变化的弯曲！")
    print(f"      （球极投影是保角映射，把 S^3 常曲率映射到 R^3 共形平坦且仍常曲率）")

    # ---- Weyl 张量（3 维恒 0）----
    print()
    print("3. Weyl 张量（引力波）：")
    print("   3 维流形的 Weyl 张量恒为 0（Weyl 只在 n>=4 非平凡）。")
    print("   => 共形平坦弯曲（3 维）没有引力波（Weyl=0）。")
    print("   引力波（Weyl != 0）需要 4 维（3+1 时空，空间+时间）。")

    print()
    print("=== 结论 ===")
    print("  球极投影下 Bures 度量 = 共形平坦弯曲（标量曲率非零，弯曲 ✅）。")
    print("  但 3 维 Weyl 恒 0 => 无引力波。")
    print("  引力波需要 4 维 Weyl，而 4 维 = 空间（球极投影）+ 时间（号差）的缝合。")
    print("  所以「共形平坦够不够」要看 4 维（空间+时间），不只是纯空间球极投影。")

    summary = {
        "conformal_factor": str(Omega),
        "scalar_curvature": str(R),
        "scalar_curvature_nonzero": bool(R != 0),
        "weyl_3d_zero": True,
        "conclusion": "stereographic Bures metric = conformally flat (scalar curvature != 0, "
                      "curved), but 3D Weyl = 0 (no gravitational waves); gravitational waves "
                      "need 4D Weyl = space (stereographic) + time (signature) splice.",
    }
    out = ROOT / "experiments" / "exp_verify_stereographic_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
