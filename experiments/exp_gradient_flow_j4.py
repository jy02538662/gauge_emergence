"""Does the restoring-force gradient flow (with topology) give time evolution J (J^4=1)?

Proposition: 回归力 = gradient flow D' = -grad S[D], + topological constraint
            -> solves D(t) -> gives J with J^4=1 -> Cayley-Dickson ladder.

Test honestly:
  1. S = Tr(D^2), gradient flow D' = -2D -> D(t) = D(0) e^{-2t} (dissipative, monotonic decay).
  2. Wick rotation (i): D(t) = D(0) e^{2 i t} (CONTINUOUS U(1), period pi, not discrete Z4).
  3. pi-flux topology: phase theta = pi (Z2), NOT Z4.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    D0 = 1.0
    # 1. gradient flow (Euclidean): D' = -2D -> D(t) = D0 e^{-2t}
    t = np.linspace(0, 3, 200)
    D_euclid = D0 * np.exp(-2.0 * t)
    # 2. Wick rotation: D(t) = D0 e^{2 i t}
    D_lorentz = D0 * np.exp(2j * t)
    # sample phases of the Wick-rotated flow at t = 0, pi/4, pi/2, ... to see if Z4
    samples = [D0 * np.exp(2j * s) for s in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]]
    # 3. pi-flux phase structure: theta = pi (Z2)
    theta_pi = np.exp(1j * np.pi)

    print("1. 梯度流 D' = -2D（欧氏）:")
    print(f"   D(3) = {D_euclid[-1]:.4f}  -> 单调衰减到 0（耗散，非循环）")
    print()
    print("2. Wick 转动 D(t) = D0 e^{2it}:")
    print(f"   |D(t)| = {abs(D_lorentz[-1]):.4f}（守恒，振荡）")
    print(f"   相位采样 t=0..pi: {[f'{complex(s):.3f}' for s in samples]}")
    print("   -> 连续 U(1) 旋转（周期 pi），不是离散 Z4（只有 4 个值）")
    print()
    print("3. pi 磁通拓扑约束: 相位 theta = pi = {+1,-1}（Z2），不是 Z4")
    print(f"   e^(i*pi) = {theta_pi:.1f}")
    print()
    print("结论：")
    print("  - 梯度流（回归力）给的是固定点（耗散松弛到 pi 磁通），不是 J^4=1 循环；")
    print("  - Wick 转动给的是连续 U(1)，不是离散 Z4；")
    print("  - 拓扑约束给的是 Z2（pi 磁通），不是 Z4；")
    print("  - 所以 J（J^2=-1, J^4=1）是独立公设（时间=有向区分），不是梯度流推出来的。")
    print("  => 动力学缺失 不被回归力闭合；J 是公设终点。")

    out = ROOT / "experiments" / "exp_gradient_flow_j4_last_run.json"
    out.write_text(json.dumps({
        "euclid_converges": bool(abs(D_euclid[-1]) < 0.01),
        "lorentz_continuous_U1": bool(abs(abs(D_lorentz[-1]) - 1.0) < 1e-9),
        "pi_flux_Z2_not_Z4": bool(np.isclose(theta_pi, -1.0)),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
