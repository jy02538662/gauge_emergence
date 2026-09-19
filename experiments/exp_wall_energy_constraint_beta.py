"""<E>（观察者能量）的定义：能量密度约束给出 beta=0 还是 beta>0？

第 8 轮的「最大熵面给紧 D」有一个隐藏假设：「最大熵 + 能量约束 <E>」。
但 <E> 的取值决定 beta：
  - 无约束（只有归一化）-> 均匀 rho=1/N -> beta=0 -> 无紧。
  - 能量约束 <E> -> 热态 rho=e^{-beta E} -> beta 由 <E> 决定。

关键问题：D（有限观察者）的能量密度 <E> = Tr(D^2)/N = 4（degree）对应 beta=0 还是 beta>0？

检验：<E>(beta) = sum E_i e^{-beta E_i} / Z（E_i = lambda_i^2），看 beta=0 时 <E> 是否 = 4（均匀平均），
beta>0 时 <E> 是否 < 4（下降）。

若 <E>(beta=0) = 4 = D 的能量密度，则 D 不逼出 beta>0（紧），
「有限观察者 -> 紧 D」在最后一步断掉。

Code: `py -m experiments.exp_wall_energy_constraint_beta`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pi_flux_D(L):
    N = L * L

    def idx(x, y):
        return (y % L) * L + (x % L)

    D = np.zeros((N, N))
    for y in range(L):
        for x in range(L):
            i = idx(x, y)
            j = idx(x + 1, y)
            w = (-1.0) ** y
            D[i, j] = w
            D[j, i] = w
            j = idx(x, y + 1)
            D[i, j] = 1.0
            D[j, i] = 1.0
    return D


def mean_energy(ev, beta):
    """<E>(beta) = sum E_i e^{-beta E_i} / sum e^{-beta E_i}, E_i = lambda_i^2."""
    E = ev ** 2
    w = np.exp(-beta * E)
    return float(np.sum(E * w) / np.sum(w))


def main():
    print("=== <E>（观察者能量）的定义：能量密度约束 -> beta=0 还是 beta>0 ===")
    print()

    L = 16
    N = L * L
    D = pi_flux_D(L)
    ev = np.linalg.eigvalsh(D)
    E = ev ** 2
    tr2_over_N = float(np.sum(E) / N)
    print(f"D = pi-flux torus (L={L}, N={N})，能量密度 Tr(D^2)/N = {tr2_over_N:.6f}")
    print(f"（= degree = 4，均匀平均 = beta=0 时的 <E>）")
    print()

    print(f"  {'beta':>8} {'<E>(beta)':>14} {'<E> vs 4':>12}")
    rows = []
    for beta in [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        mE = mean_energy(ev, beta)
        rows.append((beta, mE))
        print(f"  {beta:>8} {mE:>14.6f} {'=4(均匀)' if beta == 0 else '下降':>12}")

    print()
    print("  <E>(beta=0) = 4（均匀平均 = D 的能量密度）。")
    print("  <E>(beta>0) < 4（e^{-beta lambda^2} 压低大 lambda^2，能量下降）。")
    print("  => 能量密度约束 <E>=4 的唯一解是 beta=0（均匀，无紧）。")
    print("     要 beta>0（紧 D），需要 <E> < 4（观察者能量低于均匀平均 = 额外偏好）。")

    print()
    print("=== 结论 ===")
    print("  「最大熵面给紧 D」的隐藏假设被点破：")
    print("    最大熵（无约束）-> 均匀 beta=0 -> 无紧；")
    print("    最大熵 + 能量约束 <E> -> 热态 beta>0 -> 紧，但 <E>=4（D 的能量密度）")
    print("    恰好逼出 beta=0（无紧）。")
    print("  => 有限观察者（D 有限维）的能量密度 = 均匀平均（beta=0），不逼出 beta>0。")
    print("     「有限观察者 -> 紧 D」在最后一步断掉：紧 D 需要 <E><4（额外偏好）。")

    summary = {
        "question": "does D's energy density <E>=4 give beta=0 or beta>0?",
        "answer": "beta=0 (uniform, no compact D). <E>(beta=0)=4 = uniform average = D's energy "
                  "density. beta>0 requires <E><4 (energy below uniform average = extra "
                  "preference), which 'no-preference' does NOT give.",
        "table": [{"beta": float(b), "mean_energy": float(m)} for b, m in rows],
        "conclusion": "the hidden assumption of 'maxent face gives compact D' is exposed: "
                      "maxent (unconstrained) -> uniform beta=0 (no compact); maxent + energy "
                      "constraint <E> -> thermal beta>0 (compact), but <E>=4 (D's energy density) "
                      "gives beta=0. Finite observer's energy density = uniform average, does NOT "
                      "force beta>0. The chain 'finite observer -> compact D' breaks at the last "
                      "step: compact D needs <E><4 (extra preference).",
    }
    out = ROOT / "experiments" / "exp_wall_energy_constraint_beta_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
