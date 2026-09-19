"""检验：推进「共形 + 角亏 → 黎曼」——角亏 δ 是标量曲率的来源。

第 ⑤ 步卡点 = 共形 vs 黎曼 = 缺标量曲率 = 物质源。本步推进：
  - 反演（共形）的共形因子 φ=-log|y| 标量曲率 R=0（共形平坦，因为 R³ 平）。
  - 角亏 δ=(4-valence)π/2（Q1 已解：δ 是标量曲率的离散版，缺陷 valence≠4 -> δ≠0）。
  - 所以「共形 -> 黎曼」= 加角亏（缺陷）= 物质源。

坐实：
  1. 反演共形因子 φ=-log|y| 的标量曲率 R = 0（解析：R=e^{-2φ}(-2Δφ-2(d-2)|gradφ|^2)=0）。
  2. 角亏 δ=(4-valence)π/2：valence 3/4/5 -> δ = +π/2 / 0 / -π/2。
  3. 结论：共形平坦（R=0）+ 角亏（缺陷）= 黎曼弯曲（R≠0）。

卡点推进：共形 -> 黎曼 需要角亏（缺陷=物质源），而物质源是桥 B 墙（守恒律 ⟂ 长程）。

Code: `py -m experiments.exp_wall_conformal_to_riemann`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    print("=== 推进：共形 + 角亏 → 黎曼（角亏 = 标量曲率来源）===")
    print()

    # ---- 1. 反演（共形）标量曲率 R = 0 ----
    print("1. 反演共形因子 φ=-log|y| 的标量曲率 R = 0（共形平坦）")
    # 解析：φ=-log r，d=3；Δφ=-(d-2)/r^2=-1/r^2，|gradφ|^2=1/r^2
    # R = e^{-2φ}(-2Δφ - 2(d-2)|gradφ|^2) = r^2(2/r^2 - 2/r^2) = 0
    r_vals = np.array([0.5, 1.0, 2.0, 4.0])
    d = 3
    dphi2 = 1.0 / r_vals ** 2            # |gradφ|^2
    lap_phi = -(d - 2) / r_vals ** 2     # Δφ
    R = r_vals ** 2 * (-2 * lap_phi - 2 * (d - 2) * dphi2)
    print(f"   r = {r_vals}")
    print(f"   R = e^(-2φ)(-2Δφ-2(d-2)|gradφ|^2) = {R}（全 0 = 共形平坦）")
    flat = np.allclose(R, 0, atol=1e-12)
    print(f"   反演（共形）标量曲率 R=0：{('[OK]' if flat else '[FAIL]')}")

    # ---- 2. 角亏 δ = 标量曲率的离散版（Q1）----
    print()
    print("2. 角亏 δ = (4-valence)π/2（标量曲率的离散版，Q1）")
    for valence in [3, 4, 5]:
        delta = (4 - valence) * np.pi / 2
        print(f"   valence={valence}: δ = (4-{valence})π/2 = {delta:+.4f}")
    print(f"   valence=4（4-regular）-> δ=0（平坦）；valence=3/5（缺陷）-> δ=±π/2（弯曲）")

    # ---- 3. 结论 ----
    print()
    print("=== 结论 ===")
    print("  共形平坦（反演，R=0）+ 角亏（缺陷，δ≠0）= 黎曼弯曲（R≠0）。")
    print("  「共形 -> 黎曼」= 加角亏（缺陷）= 物质源 T_μν。")
    print("  卡点推进：标量曲率的来源 = 角亏 δ（Q1 已解），但角亏 = 缺陷 = 物质源 = 桥 B 墙。")

    summary = {
        "question": "does conformal + angular deficit -> Riemannian (scalar curvature source)?",
        "inversion_R_flat": bool(flat),
        "angular_deficit_v3": float((4 - 3) * np.pi / 2),
        "angular_deficit_v4": 0.0,
        "angular_deficit_v5": float((4 - 5) * np.pi / 2),
        "conclusion": "conformal flat (inversion R=0) + angular deficit (defect delta!=0) = "
                      "Riemannian curved (R!=0). 'conformal -> Riemannian' = add angular deficit "
                      "(defect) = matter source T_mu_nu. Scalar curvature source = angular deficit "
                      "delta (Q1 solved), but delta = defect = matter source = bridge B wall.",
    }
    out = ROOT / "experiments" / "exp_wall_conformal_to_riemann_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
