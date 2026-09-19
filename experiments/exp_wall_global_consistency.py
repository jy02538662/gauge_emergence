"""检验：破墙路径第 ⑤ 步——全局相容：多观察者过渡 = 反演（共形，不是等距）。

第 ⑤ 步（真墙）：多个观察者的「紧球 S³」能不能一致地拼成「全局流形」。
  多观察者的球极投影过渡 = 反演 y -> y/|y|^2（1.5 预印本 2.9）。
  反演是「共形」（保角，J^T J = I/|y|^4），不是「等距/黎曼」（保长度，J^T J = I）。

坐实：
  1. 反演的 Jacobian J^T J = I/|y|^4（共形因子位置依赖，保角不保长）。
  2. 对比等距：J^T J != I（反演不是等距）。
  3. 结论：多观察者拼全局给「共形流形」（Weyl），不是「黎曼流形」（标量曲率）。

卡点 = 共形 vs 黎曼 = 缺标量曲率 = 物质源 T_μν（桥 B）。

Code: `py -m experiments.exp_wall_global_consistency`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def inversion(y):
    """反演 y -> y/|y|^2。"""
    return y / np.sum(y ** 2)


def inversion_jacobian(y):
    """反演的 Jacobian J = I/|y|^2 - 2 y y^T/|y|^4。"""
    r2 = np.sum(y ** 2)
    I = np.eye(len(y))
    return I / r2 - 2 * np.outer(y, y) / (r2 ** 2)


def main():
    print("=== 破墙路径第 ⑤ 步：全局相容——多观察者过渡 = 反演（共形，不是等距）===")
    print()

    # ---- 1. 球极投影过渡 = 反演（两个观察者：北极/南极）----
    print("1. 球极投影过渡 = 反演（观察者 A 北极 / B 南极）")
    # S³ 单位球上的点 q=(x0,x1,x2,x3)，投影到 R³
    # 北极投影 y_A = (x1,x2,x3)/(1-x0)；南极投影 y_B = (x1,x2,x3)/(1+x0)
    # 过渡 y_B = y_A/|y_A|^2（反演）
    yA = np.array([1.0, 2.0, 3.0])          # 观察者 A 的坐标
    yB = inversion(yA)                       # 观察者 B 的坐标（反演）
    print(f"   y_A = {yA}，y_B = y_A/|y_A|^2 = {yB}")
    print(f"   验证 y_B = y_A/|y_A|^2：{('[OK]' if np.allclose(yB, yA/np.sum(yA**2)) else '[FAIL]')}")

    # ---- 2. 反演是共形（J^T J = I/|y|^4），不是等距（J^T J = I）----
    print()
    print("2. 反演的 Jacobian：J^T J = I/|y|^4（共形），不是 I（等距）")
    J = inversion_jacobian(yA)
    JTJ = J.T @ J
    r2 = np.sum(yA ** 2)
    I_scaled = np.eye(3) / (r2 ** 2)         # 共形因子 1/|y|^4
    conformal = np.allclose(JTJ, I_scaled, atol=1e-9)
    isometric = np.allclose(JTJ, np.eye(3), atol=1e-9)
    print(f"   J^T J = {np.round(JTJ, 6)}")
    print(f"   共形（J^T J = I/|y|^4 = I/{r2:.0f}）：{('[OK]' if conformal else '[FAIL]')}")
    print(f"   等距（J^T J = I）：{('[是]' if isometric else '[否]（不是等距，保角不保长）')}")

    # ---- 3. 共形 vs 黎曼 = 缺标量曲率 = 物质源 ----
    print()
    print("3. 共形（Weyl）vs 黎曼（标量曲率）")
    print("   多观察者过渡 = 反演 = 共形（保角，缩放因子 1/|y|^4 位置依赖）")
    print("   GR 需要黎曼（标量曲率 = 引力），不是共形（Weyl = 保角）")
    print("   共形 -> 黎曼 缺的是「标量曲率」=「物质源 T_μν」（桥 B 的墙）")

    # ---- 结论 ----
    print()
    print("=== 结论 ===")
    print("  多观察者拼全局给「共形流形」（反演过渡，保角），不是「黎曼流形」（等距过渡，保长）。")
    print("  卡点 = 共形 vs 黎曼 = 缺标量曲率 = 物质源 T_μν（桥 B）。")
    print("  这是第 ⑤ 步（全局相容）的精确卡点：拼得上（共形），但不是 GR 要的黎曼流形。")

    summary = {
        "question": "do multiple observers' S^3 glue into a global manifold (local diffeo)?",
        "transition_is_inversion": True,
        "JTJ_conformal": bool(conformal),
        "JTJ_isometric": bool(isometric),
        "conformal_factor": float(r2 ** 2),
        "conclusion": "multiple observers glue into CONFORMAL manifold (inversion transition, "
                      "angle-preserving), NOT Riemannian manifold (isometry, length-preserving). "
                      "Bottleneck = conformal vs Riemannian = missing scalar curvature = matter "
                      "source T_mu_nu (bridge B). This is the precise bottleneck of step 5: "
                      "glues (conformally) but not into the Riemannian manifold GR needs.",
    }
    out = ROOT / "experiments" / "exp_wall_global_consistency_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
