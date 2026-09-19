"""无偏好 -> 无紧是必然，紧 D 需要「热态」（能量约束）。

「无偏好」的两个读法：
  面① 尺度不变（无外部参考系）：rho = C/lambda（幂律）-> log rho 对数 -> 无紧。
  面② 最大熵（无约束）：rho = 1/N（均匀）-> log rho 常数 -> 无紧。
  只有「热态」：rho = e^{-beta E}（指数，最大熵 + 能量约束 <E>）-> 紧。

检验三个态的 |log rho|^{-1} 的 Dixmier 迹：
  幂律 1/i、均匀 1/N、指数 e^{-beta i}。

若幂律和均匀都发散（无紧），只有指数有限（紧），则：
  「无偏好」的两个读法都无紧，紧 D 需要「能量约束 <E>」（能量偏好 = 额外输入，
  不是「无偏好」逼出的）。第八轮的「最大熵面」是偷换概念（把「无偏好」从
  「尺度不变」换成「最大熵 + 能量约束」）。

Code: `py -m experiments.exp_wall_no_preference_compactness`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def partial_dixmier(mu_descending):
    N = len(mu_descending)
    return float(np.sum(mu_descending) / np.log(N))


def main():
    print("=== 无偏好 -> 无紧是必然，紧 D 需要热态（能量约束）===")
    print()

    print(f"  {'N':>9} {'幂律1/i':>12} {'均匀1/N':>12} {'指数e^-i':>14}")
    rows = []
    for N in [16, 64, 256, 1024, 4096, 16384, 65536, 262144]:
        i = np.arange(1, N + 1, dtype=float)
        # 幂律（尺度不变）rho=1/i：|log rho|^{-1} = 1/log i
        mu_power = np.sort(1.0 / np.log(i[1:]))[::-1]
        # 均匀（最大熵无约束）rho=1/N：|log rho|^{-1} = 1/log N（常数）
        mu_uniform = np.full(N, 1.0 / np.log(N))
        # 指数（热态，能量约束）rho=e^{-i}：|log rho|^{-1} = 1/i
        mu_exp = np.sort(1.0 / i)[::-1]
        t_power = partial_dixmier(mu_power)
        t_uniform = partial_dixmier(mu_uniform)
        t_exp = partial_dixmier(mu_exp)
        rows.append((N, t_power, t_uniform, t_exp))
        print(f"  {N:>9} {t_power:>12.4f} {t_uniform:>12.4f} {t_exp:>14.4f}")

    print()
    print("  幂律（尺度不变）: Dixmier 迹发散 -> 无紧。")
    print("  均匀（最大熵无约束）: Dixmier 迹发散 -> 无紧。")
    print("  指数（热态，能量约束）: Dixmier 迹 -> 1 -> 紧。")

    print()
    print("=== 结论 ===")
    print("  「无偏好」的两个读法（尺度不变幂律 + 最大熵均匀）都无紧。")
    print("  紧 D 需要「热态」（指数）= 最大熵 + 能量约束 <E>。")
    print("  而能量约束 <E>（<E><4 能量偏好）是额外输入，不是「无偏好」逼出的。")
    print("  => 第八轮的「最大熵面」是偷换概念：")
    print("     「无偏好」= 尺度不变（面①）或均匀（面②无约束），都无紧；")
    print("     「最大熵 + 能量约束」= 热态（紧），但「能量约束」是「无偏好」之外的。")
    print("  P4 确实是「无偏好」的必然：破它需要「能量偏好」或「有限探测能量」（新输入）。")

    summary = {
        "question": "do BOTH readings of 'no-preference' give no-compact-D?",
        "answer": "YES: scale-invariant (power-law 1/i) AND maxent-unconstrained (uniform 1/N) "
                  "both give divergent Dixmier trace (no compact). Only thermal (exponential "
                  "e^{-beta i}, maxent + energy constraint) gives compact.",
        "table": [{"N": int(N), "power": float(p), "uniform": float(u), "exp": float(e)}
                  for N, p, u, e in rows],
        "conclusion": "'no-preference' (both readings) -> no compact D, NECESSARILY. Compact D "
                      "needs thermal (energy constraint <E><4 = energy preference), which is an "
                      "EXTRA input not given by 'no-preference'. Round-8 'maxent face' was a "
                      "concept-swap. P4 is truly the consequence of 'no-preference'.",
    }
    out = ROOT / "experiments" / "exp_wall_no_preference_compactness_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
