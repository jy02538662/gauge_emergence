"""lambda_c（过渡尺度）的内生性：lambda_c ~ <E> = D 谱的平均能量（O(1) 内生常数）。

链条：
  无偏好（最大熵面）+ 有限观察者（有限能量 <E>）-> 热态 rho = e^{-beta E}
  -> 特征尺度 lambda_c = 1/beta。而 <E> = D 谱的平均能量（D 自己长出的）。
  => 若 <E> 是 O(1) 内生常数，则 lambda_c 内生（不手放）。

检验：D = pi-flux torus 的
  1. Tr(D^2)/N = sum lambda_i^2 / N（谱作用量能量密度，解析 = 4 = degree）；
  2. sum |lambda_i| / N（平均谱绝对值，数值算，应 O(1) 收敛）。

两者随 N 的标度 = lambda_c 内生性的签名。

Code: `py -m experiments.exp_wall_lambdac_intrinsic`
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


def main():
    print("=== lambda_c 内生性：D 谱的平均能量 <E> 随 N 的标度 ===")
    print()

    print(f"  {'L':>5} {'N=L^2':>8} {'Tr(D^2)/N':>12} {'sum|lambda|/N':>14}")
    rows = []
    for L in [4, 8, 16, 32, 64]:
        N = L * L
        D = pi_flux_D(L)
        # 1. Tr(D^2)/N = sum |D_ij|^2 / N（= degree = 4，解析/快速累加，避免稠密 D@D）
        tr2_over_N = float(np.sum(np.abs(D) ** 2) / N)
        # 2. sum |lambda_i| / N（平均谱绝对值，数值算谱，N<=4096 时 eigh）
        if N <= 4096:
            ev = np.linalg.eigvalsh(D)
            avg_abs = float(np.sum(np.abs(ev)) / N)
        else:
            avg_abs = float('nan')
        rows.append((L, N, tr2_over_N, avg_abs))
        print(f"  {L:>5} {N:>8} {tr2_over_N:>12.6f} {avg_abs:>14.6f}")

    print()
    print("  Tr(D^2)/N = 4（精确 = degree，内生常数，不随 N 变）。")
    print("  sum|lambda|/N ~ O(1)（平均谱绝对值，收敛到有限常数）。")
    print("  => <E>（D 谱的平均能量）是 O(1) 内生常数，不随 N 发散。")

    print()
    print("=== 结论 ===")
    print("  lambda_c ~ 1/beta ~ <E>（最大熵高温近似 beta ~ 1/<E>）。")
    print("  <E> = O(1) 内生常数（D 的 degree=4 或平均谱绝对值）。")
    print("  => lambda_c 内生（不手放）：过渡尺度 = D 谱的平均能量。")
    print("     有限观察者（D 有限维）+ 最大熵 -> 温度 beta ~ 1/<E> -> lambda_c ~ <E>。")

    summary = {
        "question": "is lambda_c (crossover scale) intrinsic to finite observer D?",
        "answer": "YES (heuristic): lambda_c ~ 1/beta ~ <E> = average spectral energy of D, "
                  "which is O(1) intrinsic (Tr(D^2)/N = degree = 4, independent of N).",
        "table": [{"L": int(L), "N": int(N), "TrD2_over_N": float(t), "avg_abs": float(a)}
                  for L, N, t, a in rows],
        "conclusion": "lambda_c ~ <E> = O(1) intrinsic (D's degree / average spectral energy). "
                      "Finite observer (finite-dim D) + maxent -> temperature beta ~ 1/<E> -> "
                      "lambda_c ~ <E>, NOT hand-placed.",
        "honest_boundary": "beta ~ 1/<E> is high-temperature approximation; energy definition "
                           "E=|lambda| or lambda^2 is a choice; 'maxent' reading is itself a "
                           "concept choice (8th round). But lambda_c ~ O(1) intrinsic is robust.",
    }
    out = ROOT / "experiments" / "exp_wall_lambdac_intrinsic_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
