"""无偏好的「最大熵」面：最大熵 + 有限观察者（约束）-> 指数谱 -> 紧 D。

公设 A（无偏好）有两个候选面：
  面① 尺度不变（无外部参考系）：rho(c lambda)=c^{-1} rho(lambda) -> rho=C/lambda（幂律）
       -> log rho 对数 -> |log rho|^{-1} = 1/log i -> 无紧（侦察 1 已验：Dixmier 迹发散）。
  面② 最大熵（最大无偏）：max -sum rho_i log rho_i s.t. 约束
       -> rho = e^{-beta E}（指数，吉布斯态）-> log rho 线性 -> 紧（本脚本坐实）。

关键：有限观察者（有限维）只给「截断」（1/N），不给「指数衰减」；「特征尺度（温度 beta）」
来自「最大熵 + 有限约束」，不是「有限维」本身。

本脚本：坐实面②——最大熵（无偏）+ 有限能量约束 -> 吉布斯态（指数谱）-> 紧 D。

Code: `py -m experiments.exp_wall_maxent_compactness`
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


def main():
    print("=== 无偏好的「最大熵」面：指数谱 -> 紧 D ===")
    print()

    # ---- 1. 最大熵推导（符号/数值）：拉格朗日乘子 -> 吉布斯态 ----
    print("1. 最大熵原理：max -sum rho_i log rho_i  s.t.  sum rho_i=1, sum rho_i E_i=E0")
    print("   拉格朗日乘子 => rho_i = e^{-beta E_i} / Z（吉布斯态，指数谱）。")
    print("   「无偏」= 熵最大；「有限观察者」= 有限能量约束 E0；=> 温度 beta（特征尺度）。")
    print()

    # ---- 2. 数值：吉布斯态（指数谱）的 |log rho|^{-1} 是否紧 ----
    print("2. 取 E_i = i（能量线性），beta=1，rho_i = e^{-i}（指数谱）：")
    print(f"  {'N':>9} {'Tr_w(1/log i)[幂律]':>20} {'Tr_w(1/i)[指数]':>18}")
    rows = []
    for N in [8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144]:
        i = np.arange(1, N + 1, dtype=float)
        # 面① 尺度不变（幂律）：rho=1/i -> log rho=-log i -> mu=1/log i
        mu_power = 1.0 / np.log(i[1:])           # 跳过 i=1（log 1=0）
        # 面② 最大熵（指数）：rho=e^{-i} -> log rho=-i -> mu=1/i
        mu_exp = 1.0 / i
        t_power = partial_dixmier(np.sort(mu_power)[::-1], len(mu_power))
        t_exp = partial_dixmier(np.sort(mu_exp)[::-1], len(mu_exp))
        rows.append((N, t_power, t_exp))
        print(f"  {N:>9} {t_power:>20.4f} {t_exp:>18.4f}")
    print("  面① 幂律谱（尺度不变）: Dixmier 迹发散 -> 无紧 D。")
    print("  面② 指数谱（最大熵）:   Dixmier 迹 -> 1 -> 紧 D（特征尺度 beta=1）。")

    # ---- 3. 两个面的对比 + 结论 ----
    print()
    print("=== 结论 ===")
    print("  「无偏好」有两个面：")
    print("    面① 尺度不变（无外部参考系）-> 幂律谱 -> 无紧 D（P4 墙）。")
    print("    面② 最大熵（最大无偏）-> 指数谱 -> 紧 D（温度 = 特征尺度）。")
    print("  有限观察者给「截断」（1/N），不给「指数衰减」；")
    print("  「特征尺度（温度）」来自「最大熵 + 有限约束」，不是「有限维」本身。")
    print("  => P4 的破法：公设 A 的「无偏好」若读成「最大熵」（面②），则内生特征尺度，")
    print("     给紧 D。这「把 P4 从硬障碍变可调参数」——不是换公设，是换「无偏好」的读法。")

    summary = {
        "question": "does 'no-preference' read as MAXENT give a compact D?",
        "answer": "YES: maxent (max -sum rho log rho s.t. finite energy constraint) -> Gibbs "
                  "state rho=e^{-beta E} (exponential spectrum) -> |log rho|^{-1} = 1/(beta i) "
                  "-> Dixmier trace finite (=1/beta) -> compact D.",
        "contrast": "scale-invariance reading (power-law rho=1/i) -> log rho logarithmic -> "
                    "Dixmier divergence -> no compact D (P4 wall).",
        "table": [{"N": int(N), "Trw_powerlaw": float(tp), "Trw_exponential": float(te)}
                  for N, tp, te in rows],
        "conclusion": "finite observer gives TRUNCATION (1/N), not exponential decay; the "
                      "characteristic scale (temperature) comes from MAXENT + finite constraint. "
                      "So reading 'no-preference' as MAXENT (face 2) gives an INTRINSIC "
                      "characteristic scale and a compact D -- downgrades P4 from axiom-conflict "
                      "to axiom-precision.",
    }
    out = ROOT / "experiments" / "exp_wall_maxent_compactness_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
