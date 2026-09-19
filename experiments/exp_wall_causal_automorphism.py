"""检验：'互联 = 因果 = 时间（有向区分）' 下，保因果自同构 = 保序置换 = 小离散群。

对着靶点 Aut_互联(R)/Inn(R) = Diff(M) 的有限维侧面：
  - 有限维 Aut(M_n) = Inn(M_n)（Skolem–Noether，见 exp_wall_skolem_noether），
    所以「保因果自同构」= {保因果的酉共轭} = {保因果的置换}。
  - 因果结构 = 时间（有向区分）给的偏序 ≺。
  - 保因果置换 = 保 ≺ 的置换，是一个**小离散群**（保序置换），远小于 Diff(M)。

三个检验（离散因果结构 -> 保因果置换群的大小）：
  1. 反链（纯空间，无因果序）：保因果 = 全置换 S_N（最大，无约束）。
  2. 1+1 维光锥网格：保因果 = 小离散群（中间）。
  3. 1D 全序（时间链）：保因果 = 平凡 {e}（最小，严格保序只有恒等）。

对比（陈述，非数值）：
  连续保因果（Malament）= 共形群（无限维连续）；离散保因果 = 保序置换（有限小群）。

结论：离散保因果 = 保序置换（小、离散）；连续保因果 = 共形群（大、连续）。
  Diff(M) 的正确刻画 = 「因果 + 连续极限」，而「连续极限」= 光滑化 = Connes 核心。
  => 「互联=因果」靶点没绕开墙，而是把墙精确到「离散保序 -> 连续共形」的极限。

Code: `py -m experiments.exp_wall_causal_automorphism`
"""

from __future__ import annotations

import json
import sys
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def count_causal_automorphisms(N, P):
    """数保偏序 P 的置换个数（early-exit）。"""
    count = 0
    reps = []
    for perm in permutations(range(N)):
        ok = True
        for p in range(N):
            Pp = P[p]
            sp_ = perm[p]
            for q in range(N):
                if P[sp_][perm[q]] != Pp[q]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            count += 1
            if len(reps) < 8:
                reps.append(perm)
    return count, reps


def main():
    print("=== 保因果自同构 = 保序置换 = 小离散群（有限维侧面）===")
    print()

    results = {}

    # ---- 1. 反链（纯空间，无因果序）：保因果 = 全置换 ----
    print("1. 反链（纯空间，无因果序）：保因果 = 全置换 S_N")
    N = 4
    P_anti = [[False] * N for _ in range(N)]  # 无边，无因果序
    c1, _ = count_causal_automorphisms(N, P_anti)
    print(f"   N={N}：保因果置换数 = {c1}（全置换 = {N}!，无约束，最大）")
    results["antichain_N4"] = {"count": c1, "N_factorial": 24}

    # ---- 2. 1+1 维光锥网格：保因果 = 小群 ----
    print()
    print("2. 1+1 维光锥网格：保因果 = 小离散群")
    for L in [2, 3]:
        N = L * L

        def causal(p, q):
            t, x = divmod(p, L)
            t2, x2 = divmod(q, L)
            return (t2 - t) > abs(x2 - x)

        P = [[causal(p, q) for q in range(N)] for p in range(N)]
        c2, reps = count_causal_automorphisms(N, P)
        import math
        print(f"   L={L}（{N} 事件）：保因果置换数 = {c2}（全置换 = {math.factorial(N)}，保因果 << 全置换）")
        print(f"     代表元 = {[tuple(r) for r in reps]}")
        results[f"lightcone_L{L}"] = {"N": N, "count": c2,
                                      "N_factorial": math.factorial(N),
                                      "reps": [str(tuple(r)) for r in reps]}

    # ---- 3. 1D 全序（时间链）：保因果 = 平凡 ----
    print()
    print("3. 1D 全序（时间链 i<j）：保因果 = 平凡 {e}")
    N = 4
    P_total = [[i < j for j in range(N)] for i in range(N)]
    c3, reps = count_causal_automorphisms(N, P_total)
    print(f"   N={N}：保因果置换数 = {c3}（严格保序只有恒等）")
    print(f"     代表元 = {[tuple(r) for r in reps]}")
    results["total_order_N4"] = {"count": c3, "reps": [str(tuple(r)) for r in reps]}

    # ---- 结论 ----
    print()
    print("=== 结论 ===")
    print("  离散因果结构：反链 -> S_N（最大） -> 光锥 -> 小群 -> 全序 -> {e}（最小）。")
    print("  因果结构越强，保因果置换群越小（对称性被「序」冻结）。")
    print("  连续保因果（Malament）= 共形群（无限维）；离散保因果 = 保序置换（有限）。")
    print("  => Diff(M) 的正确刻画 = '因果 + 连续极限'，'连续极限' = 光滑化 = Connes 核心。")
    print("  => '互联=因果' 靶点把墙精确到「离散保序 -> 连续共形」的极限，没绕开墙。")

    summary = {
        "question": "under 'interconnectedness = causality = time (directed distinction)', "
                    "what is the causal-preserving automorphism group (finite-dim)?",
        "answer": "causal-preserving automorphisms = order-preserving permutations = SMALL "
                  "discrete group, far smaller than Diff(M). Antichain -> S_N (max), "
                  "lightcone -> small, total order -> {e} (min).",
        "results": results,
        "conclusion": "discrete causal-preserving = order-preserving permutation (small, discrete); "
                      "continuous causal-preserving = conformal group (Malament, large, continuous). "
                      "Diff(M) = 'causality + continuous limit', and the continuous limit = "
                      "smoothing = Connes core. 'interconnectedness=causality' pinpoints the wall "
                      "as 'discrete order-preserving -> continuous conformal', does not bypass it.",
    }
    out = ROOT / "experiments" / "exp_wall_causal_automorphism_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
