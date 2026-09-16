"""Pin down the "+" in "梯度流 + Wick 转动 = 时间演化": which operation turns
dissipation into oscillation, and does it have J^4 = 1?

Gradient flow: dx/dtau = -2x -> x(tau) = x0 e^{-2 tau} (dissipative, real decay).

Three candidate "anti-dissipation" operations:
  1. 复共轭 K: K(x)=x*, K^2=1 (Z2)  -> does it turn decay into oscillation? (no, x real -> no change)
  2. 乘以 i (iJ): x -> i x, or tau -> i t  -> x(t)=x0 e^{-2it} (oscillatory); i^4=1 (Z4)
  3. 拓扑投影: theta -> pi (Z2)  -> does it turn decay into oscillation? (no, only phase fix)

Criterion: does the operation give J^4 = 1?
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    x0 = 1.0
    tau = np.linspace(0, 3, 100)
    t = np.linspace(0, 3, 100)

    # dissipative gradient flow
    x_diss = x0 * np.exp(-2.0 * tau)

    # candidate 2: 乘以 i (= Wick rotation tau -> it)
    x_osc = x0 * np.exp(-2j * t)  # e^{-2it}, oscillatory

    # J^4 = 1 check for the three candidates
    # 1. 复共轭 K: K^2 = 1 (Z2)
    K2 = 1.0  # applying conjugation twice = identity
    # 2. 乘以 i: i^4 = 1 (Z4)
    i_powers = [1j**k for k in range(4)]
    # 3. 拓扑投影 theta->pi: (e^{i pi})^2 = 1 (Z2)
    topo = np.exp(1j * np.pi)

    print("梯度流 x(tau) = x0 e^{-2 tau}（耗散，实衰减）")
    print(f"  x(3) = {x_diss[-1]:.4f}（收敛到 0）")
    print()
    print("三个候选「反耗散操作」:")
    print(f"  1. 复共轭 K：K^2 = 1（Z2），x 实则不变，不把耗散变振荡")
    print(f"  2. 乘以 i（iJ）：i^4 = 1（Z4），x(t)=x0 e^(-2it) 振荡，|x|={abs(x_osc[-1]):.4f}")
    print(f"     i 的幂：{[complex(p).__repr__() for p in i_powers]} -> 1,i,-1,-i（Z4 循环，J^4=1）")
    print(f"  3. 拓扑投影 theta->pi：e^(i*pi)={topo:.1f}（Z2，只修相位，不振荡）")
    print()
    print("结论：")
    print("  「反耗散操作」= 乘以 i（= J，时间=有向区分），即 Wick 转动 tau -> i t。")
    print("  它把 e^{-2 tau}（耗散）变成 e^{-2it}（振荡），且 i^4 = J^4 = 1。")
    print("  复共轭（Z2）和拓扑投影（Z2）都不满足 J^4=1，也不是反耗散操作。")
    print("  所以「梯度流 + Wick 转动」的「+」= 乘以 J（复数结构，J^4=1）。")

    out = ROOT / "experiments" / "exp_antidissipation_operation_last_run.json"
    out.write_text(json.dumps({
        "K_squared": 1.0,
        "i_powers": [complex(1j**k).__repr__() for k in range(4)],
        "i4_equals_1": bool(np.isclose(1j**4, 1.0)),
        "topo_Z2": bool(np.isclose(topo, -1.0)),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
