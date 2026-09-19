"""建议1（P4 上界）：|log rho|^{-1} 的 Dixmier 迹发散是否依赖 rho 的谱形式。

实验 A 坐实：rho = diag(1/i)（幂律谱，尺度不变）-> |log rho|^{-1} 谱 = 1/log i，
Dixmier 迹发散 ~ N/(ln N)^2（谱连续化 = II_1 无紧 D 的数值签名）。

本脚本检验：换 rho 的谱形式，Dixmier 迹标度变不变？
  - 幂律谱   lambda_i = 1/i          （尺度不变，观察者态）-> mu_n = 1/log n -> 发散
  - 指数谱   lambda_i = e^{-i}       （温度态，有特征尺度）-> mu_n = 1/i     -> 有限（=1）
  - 拉伸指数 lambda_i = e^{-i^alpha} -> mu_n = 1/i^alpha -> alpha 决定收敛/发散

关键区分：
  尺度不变（幂律谱）-> 无紧 D（发散）；有特征尺度（指数谱）-> 有紧 D（有限）。
  => 无紧 D 是「尺度不变」的代价，不是结构性（换指数谱就有紧 D）。

Code: `py -m experiments.exp_wall_p4_rho_dependence`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def partial_dixmier(mu_descending, N):
    return float(np.sum(mu_descending) / np.log(N))


def mu_from_loglambda(log_lambda):
    """|log rho|^{-1} 的奇异值（降序）。log_lambda = log(rho 对角元)，直接传对数谱避免 exp/log 下溢。"""
    mu = 1.0 / np.abs(log_lambda)
    mu = mu[np.isfinite(mu) & (mu > 0)]
    return np.sort(mu)[::-1]


def main():
    print("=== 建议1（P4 上界）：Dixmier 迹发散是否依赖 rho 谱形式 ===")
    print()

    alphas = [0.5, 1.0, 2.0]  # 拉伸指数 e^{-i^alpha}
    print(f"  {'N':>9} {'幂律1/i':>10} {'指数e^-i':>11} "
          + "".join([f"{'e^-i^'+str(a):>12}" for a in alphas]))
    rows = {}
    for N in [64, 256, 1024, 4096, 16384, 65536, 262144]:
        i = np.arange(1, N + 1, dtype=float)
        # 幂律谱（尺度不变）: log lambda_i = -log i
        mu_power = mu_from_loglambda(-np.log(i))
        # 指数谱（温度态）: log lambda_i = -i（直接用，避免 exp 下溢）
        mu_exp = mu_from_loglambda(-i)
        # 拉伸指数: log lambda_i = -i^alpha
        mu_stretch = [mu_from_loglambda(-(i ** a)) for a in alphas]
        t_power = partial_dixmier(mu_power, len(mu_power))
        t_exp = partial_dixmier(mu_exp, len(mu_exp))
        t_stretch = [partial_dixmier(m, len(m)) for m in mu_stretch]
        print(f"  {N:>9} {t_power:>10.3f} {t_exp:>11.4f} "
              + "".join([f"{t:>12.4f}" for t in t_stretch]))
        rows[N] = {"power": t_power, "exp": t_exp, "stretch": t_stretch}

    print()
    print("  幂律谱（尺度不变）: Dixmier 迹发散 -> 无紧 D。")
    print("  指数谱（温度态）:   Dixmier 迹 -> 1 -> 有紧 D。")
    print("  拉伸指数:           alpha>1 收敛(=0), alpha=1 ->1, alpha<1 发散。")

    print()
    print("=== 结论 ===")
    print("  无紧 D（P4）是「尺度不变（幂律谱）」的代价，不是结构性：")
    print("  换指数谱（有特征尺度）|log rho|^{-1} = 1/i 就有紧 D（Dixmier 迹=1）。")
    print("  观察者态 rho=C/lambda 是幂律谱（无偏好 A），所以无紧 D 是「无偏好」的必然。")

    summary = {
        "question": "does |log rho|^{-1} Dixmier divergence depend on rho spectrum?",
        "answer": "YES: scale-invariant (power-law 1/i) -> divergence (no compact D); "
                  "exponential e^{-i} (characteristic scale) -> finite (=1, compact D)",
        "table": {str(N): v for N, v in rows.items()},
        "conclusion": "no-compact-D (P4) is the COST of scale invariance (power-law rho=C/lambda, "
                      "no-preference A), not structural. Exponential spectrum (temperature state) "
                      "gives compact |log rho|^{-1} = 1/i (Dixmier=1).",
    }
    out = ROOT / "experiments" / "exp_wall_p4_rho_dependence_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
