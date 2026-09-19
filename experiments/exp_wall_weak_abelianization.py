"""建议2（弱交换化，对着 P1）：平均算符的交换子 ||[a_n,b_n]|| 是否随 N 衰减。

弱交换化：||[a,b]|| -> 0（N -> inf）但不等于 0。若成立，P1（无 character）的「真墙」有裂缝。

构造（超有限 II_1 = M_{2^n} 归纳极限）：
  a_n = (1/n) sum_{k=1..n} sigma_x^{(k)}  （第 k 个 M_2 因子的 sigma_x）
  b_n = (1/n) sum_{k=1..n} sigma_y^{(k)}
  [a_n, b_n] = (2i/n^2) sum_k sigma_z^{(k)}，谱范数 = 2/n -> 0。

数值预期：||[a_n,b_n]|| = 2/n（幂律 alpha=1，弱交换化成立）。

诚实边界：「平均」(1/n)sum 是手放的，对应条件期望 E（论文 #17「观察 = 条件期望」的有限维
对应），不是自反性内生。所以弱交换化「成立」，但「平均 = 条件期望」这一步需概念澄清。

Code: `py -m experiments.exp_wall_weak_abelianization`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SIGMA = {
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def sigma_at(n, k, s):
    """sigma_s 作用在第 k 个 M_2 因子上（共 n 个因子）。"""
    mats = [np.eye(2, dtype=complex)] * n
    mats[k] = SIGMA[s]
    M = mats[0]
    for m in mats[1:]:
        M = np.kron(M, m)
    return M


def main():
    print("=== 建议2（弱交换化，对着 P1）：平均算符交换子衰减 ===")
    print()

    print(f"  {'n':>4} {'dim=2^n':>8} {'||[a_n,b_n]||':>14} {'2/n':>10} {'ratio':>8}")
    rows = []
    for n in [2, 4, 6, 8, 10, 12]:
        a = sum(sigma_at(n, k, "x") for k in range(n)) / n
        b = sum(sigma_at(n, k, "y") for k in range(n)) / n
        comm = a @ b - b @ a
        norm = float(np.linalg.norm(comm, ord=2))
        rows.append((n, norm))
        print(f"  {n:>4} {2**n:>8} {norm:>14.6f} {2.0/n:>10.6f} {norm/(2.0/n):>8.4f}")

    print()
    print("  ||[a_n,b_n]|| = 2/n -> 0：弱交换化成立（幂律 alpha=1）。")
    print("  => P1 的「真墙」有裂缝：平均算符在大 N 极限交换。")

    print()
    print("=== 结论 ===")
    print("  弱交换化（平均算符交换子 -> 0）在有限维数值上成立。")
    print("  但「平均」(1/n)sum 是手放的，对应条件期望 E（论文 #17），")
    print("  不是自反性内生。弱交换化「成立」，但「平均=条件期望」这一步需概念澄清。")

    summary = {
        "question": "does ||[a_n,b_n]|| -> 0 (weak abelianization)?",
        "answer": "YES: ||[a_n,b_n]|| = 2/n -> 0 (power law alpha=1), averaged operators "
                  "a_n=(1/n)sum sigma_x^(k), b_n=(1/n)sum sigma_y^(k)",
        "table": [{"n": int(n), "dim": 2**n, "comm_norm": float(nm)} for n, nm in rows],
        "honest_boundary": "averaging (1/n)sum is hand-placed, corresponds to conditional "
                           "expectation E (paper #17), not intrinsic to self-reflexivity. "
                           "Weak abelianization HOLDS, but 'averaging = conditional expectation' "
                           "needs concept clarification.",
    }
    out = ROOT / "experiments" / "exp_wall_weak_abelianization_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
