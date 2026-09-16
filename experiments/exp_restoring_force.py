"""动力 = 回归力（梯度流）+ Wick 转动 = 时间演化？

User's insight: 动力 = 割裂之后回归的力量 (restoring force).
Math: 回归力 = gradient flow -grad S[D]; topology prevents full return -> "circling" = time evolution.

Key check: is the gradient flow (dissipative) the SAME as time evolution (unitary)?
Answer: NO directly. But gradient flow + Wick rotation (i) = time evolution.
This is the theory's "时间 = 有向区分 = J = i" (Wick rotation gamma^0 -> i gamma^0).

Verify: S(x) = x^2 (deviation from equilibrium x=0).
  - gradient flow (Euclidean): dx/dtau = -2x  -> x = x0 e^{-2 tau}  (dissipative, returns to 0)
  - Wick rotation tau = i t: dx/dt = 2i x     -> x = x0 e^{2 i t}    (oscillatory, |x| conserved)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    x0 = 1.0
    # Euclidean gradient flow: dx/dtau = -dS/dx = -2x
    tau = np.linspace(0, 3, 100)
    x_euclid = x0 * np.exp(-2.0 * tau)
    # Wick rotation tau = i t: dx/dt = 2 i x -> x = x0 e^{2 i t}
    t = np.linspace(0, 3, 100)
    x_lorentz = x0 * np.exp(2j * t)

    print("S(x) = x^2, deviation from equilibrium at x=0.  x0 = 1")
    print()
    print("1. 回归力 = 梯度流（欧氏，耗散）:  dx/dtau = -dS/dx = -2x")
    print(f"   x(3) = {x_euclid[-1]:.4f}  -> 回到平衡 x=0（耗散，单调回归）")
    print()
    print("2. Wick 转动 tau = i t（i = 时间方向 J）:  dx/dt = 2i x")
    print(f"   |x(t)| = {abs(x_lorentz[-1]):.4f}  -> 守恒（振荡，不耗散）= 时间演化")
    print()
    print("=> 回归力（梯度流）+ Wick 转动（i） = 时间演化。")
    print("   i = J = 时间 = 有向区分（理论已有，Wick 转动 gamma^0 -> i gamma^0）。")
    print("   所以「动力学缺失」= 梯度流还没被 Wick 转动成时间演化 = 桥 A 时间纳入的缺口。")

    out = ROOT / "experiments" / "exp_restoring_force_last_run.json"
    out.write_text(json.dumps({
        "euclid_converges": bool(abs(x_euclid[-1]) < 0.01),
        "lorentz_conserves": bool(abs(abs(x_lorentz[-1]) - 1.0) < 1e-9),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
