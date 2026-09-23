"""Is the octonion (strong non-associative) or a fusion category (weak/coherent)?

命题（2026-09-21，查"二阶结构"线索后）：
  「因果 = 顺序敏感」的正确数学载体是什么？两个候选：
   - 弱结合（F 符号 / fusion category）：associator 是同构、满足五边形方程（coherent）；
   - 强非结合（八元数）：(xy)z != x(yz)，不同括号给出真正不同的、无联系的元素。
  本脚本坐实：八元数是「强非结合（不 coherent）」——对 n 个元素，不同括号方式给出
  多个不同结果（不是"同构地唯一"）。对比四元数 H（结合）给出 1 个。

  结论：八元数不是 fusion category（它连"coherent 重括号"都做不到），所以
  「因果 = 顺序」的正确落点是 F 符号（弱结合），不是八元数（强非结合）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def conj(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    if n == 1:
        return x.copy()
    m = n // 2
    out = np.empty_like(x)
    out[:m] = conj(x[:m])
    out[m:] = -x[m:]
    return out


def mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    if n == 1:
        return x * y
    m = n // 2
    a, b = x[:m], x[m:]
    c, d = y[:m], y[m:]
    ac = mul(a, c)
    db = mul(conj(d), b)
    da = mul(d, a)
    bc = mul(b, conj(c))
    return np.concatenate([ac - db, da + bc])


def all_parenthesizations(elements, mul):
    """All full parenthesizations of a list of elements, returned as vectors."""
    if len(elements) == 1:
        return [elements[0]]
    out = []
    for i in range(1, len(elements)):
        lefts = all_parenthesizations(elements[:i], mul)
        rights = all_parenthesizations(elements[i:], mul)
        for l in lefts:
            for r in rights:
                out.append(mul(l, r))
    return out


def distinct_count(results, atol=1e-9):
    """Number of distinct vectors among results (clustering by allclose)."""
    reps = []
    for r in results:
        found = False
        for s in reps:
            if np.allclose(r, s, atol=atol):
                found = True
                break
        if not found:
            reps.append(r)
    return len(reps)


def main() -> None:
    rng = np.random.default_rng(0)
    dims = {"H": 4, "O": 8}

    print("n-element product: how many DISTINCT results do the Catalan parenthesizations give?")
    print("(weak/coherent => 1 distinct up to isomorphism; strong non-assoc => many distinct)\n")

    result = {}
    for name, dim in dims.items():
        for n in [3, 4, 5]:
            # pick n random elements (mix of basis vectors so it's not degenerate)
            elems = [rng.normal(size=dim) for _ in range(n)]
            # make first two basis-like to avoid accidental cancellation
            elems[0] = np.eye(dim)[1]
            elems[1] = np.eye(dim)[2] if dim >= 4 else np.eye(dim)[1]
            results = all_parenthesizations(elems, mul)
            dc = distinct_count(results)
            n_catalan = len(results)
            print(f"  {name} (dim {dim}), n={n} elements: {n_catalan} parenthesizations -> {dc} distinct result(s)")
            result[f"{name}_n{n}"] = {"n_parenthesizations": n_catalan, "n_distinct": dc}

    print("\n  => H: 1 distinct (associative/coherent).  O: >1 distinct (strong non-associative, NOT coherent).")

    out = ROOT / "experiments" / "exp_associator_pentagon_last_run.json"
    out.write_text(json.dumps({
        **result,
        "conclusion": "octonion is strong non-associative (not coherent), hence NOT a fusion category; the coherent 'order' carrier is the F-symbol (weak associativity)",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
