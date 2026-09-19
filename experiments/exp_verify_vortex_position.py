"""验证「涡旋关系 → 物理位置（定义）」是否自洽。

关键问题：Connes 距离的「局部结构」给的是「态空间度规」（Bures）还是「物理度规」？
涡旋分布 rho_v(x) 的 x 是「离散标签」还是「连续位置」？

Code: `py -m experiments.exp_verify_vortex_position`
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


def main():
    print("=== 验证「涡旋关系 -> 物理位置（定义）」是否自洽 ===")
    print()

    # ---- 1. Connes 距离的局部结构 = 态空间度规（Bures），不是物理度规 ----
    print("1. Connes 距离的「局部结构」= 态空间度规（Bures）：")
    # 单比特态空间 = Bloch 球，Bures 度量 ds^2 = (1/4)[dr^2/(1-r^2) + r^2 dOmega^2]
    # 代换 r=sin(chi) -> (1/4)[dchi^2 + sin^2 chi dOmega^2] = S^3 半球，常曲率 K=4
    # 这是「态空间的度规」，号差是 +++++（正定），不是物理时空的号差 -+++（洛伦兹）
    chi, theta, phi = sp.symbols('chi theta phi', real=True)
    # 态空间（Bures）度规：正定（3 维全是 +）
    g_bures = sp.Matrix([[sp.Rational(1, 4), 0, 0],
                         [0, sp.Rational(1, 4) * sp.sin(chi) ** 2, 0],
                         [0, 0, sp.Rational(1, 4) * sp.sin(chi) ** 2 * sp.sin(theta) ** 2]])
    ev_bures = g_bures.eigenvals()
    print(f"   态空间 Bures 度规本征值（全正 = 黎曼正定，非洛伦兹）:")
    for ev in ev_bures:
        print(f"     {ev}  (符号: {'正' if ev.is_positive else '待定'})")

    # 物理时空（闵可夫斯基）度规：号差 -+++（一个负 + 三个正）
    g_minkowski = sp.diag(-1, 1, 1, 1)
    ev_mink = g_minkowski.eigenvals()
    print(f"   物理时空（闵可夫斯基）度规本征值 = {ev_mink}  （号差 -+++，一个负）")
    print(f"   => 态空间度规（正定）≠ 物理度规（洛伦兹）。Connes 距离局部结构给的是")
    print(f"      态空间度规，不是物理度规——需要「态空间 = 物理空间」识别。")

    # ---- 2. 涡旋（投影）是离散的：rho_v(x) 的 x 是离散标签 ----
    print()
    print("2. 涡旋（投影 p）是离散的：")
    # 在有限维 D 里，投影 p 的「迹」tau(p) 是离散的（整数格）
    # 投影格（投影的集合）是离散的（量子逻辑），不是连续流形
    # 所以「涡旋分布 rho_v(x)」的 x 是「投影索引」（离散标签），不是连续位置
    print("   投影格（投影的集合）是离散的量子逻辑，不是连续流形。")
    print("   => rho_v(x) 的 x 是「离散标签」（投影索引），不是「连续位置」。")
    print("   要「连续位置」，需要「投影的连续族」= SU(2) 轨道 = S^2（内部空间），")
    print("   但 SU(2) 轨道是「内部空间」，不是「物理空间」。")

    print()
    print("=== 结论 ===")
    print("  「定义」路线（物理位置 := 涡旋关系）不自洽：")
    print("  1. rho_v(x) 预设了 x（位置）——循环（拿位置定义位置）。")
    print("  2. Connes 距离局部结构 = 态空间度规（正定），不是物理度规（洛伦兹）——")
    print("     需要「态空间 = 物理空间」识别。")
    print("  3. 涡旋（投影）是离散的，x 是离散标签，要连续位置需 SU(2) 轨道（内部空间）")
    print("     ——还是「内部 → 物理」的识别。")

    summary = {
        "bures_metric_signature": "positive-definite (Riemannian, not Lorentzian)",
        "minkowski_signature": "-+++ (one negative)",
        "vortex_projection_discrete": True,
        "rho_v_x_loop": "rho_v(x) presupposes x (position) -> circular definition",
        "conclusion": "the 'definition' route is not self-consistent: it either "
                      "circularly presupposes position, or needs 'state-space = physical "
                      "space' identification (Connes distance is a state-space metric).",
    }
    out = ROOT / "experiments" / "exp_verify_vortex_position_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
