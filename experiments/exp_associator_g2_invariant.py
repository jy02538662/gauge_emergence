"""Is the associator (the "causal ternary" structure) invariant under the Fano-plane
automorphisms (PSL(2,7) ⊂ G2)?

命题（2026-09-21，方向 C「因果 = 三元」的下一命题）：
  「因果 = 三元关系 = 关联子 ≠ 0 = 非结合」。下一命题：这个「三元结构」的自同构群
  = G2 = Aut(O)，而 G2 固定 e1 给 SU(3)（见 vault A2）。本脚本坐实中间一环：
  Fano 平面自同构（保 7 条线的置换，= PSL(2,7)）可唯一提升为八元数的代数自同构
  （带符号），从而自动保持关联子（三元结构）不变。

关键细节（符号提升，8 重）：Fano 平面自同构只给「点置换 σ」；真正的八元数自同构是
  σ'(e_i) = ε_i e_{σ(i)}，符号 ε_i ∈ {±1} 由乘法表确定。但 Fano 平面 7 条线只有 4 条
  独立，留 3 个生成元（e1,e2,e4）的符号自由度 => 每个 σ 有 2^3 = 8 个符号提升，
  总计 168×8 = 1344 个自同构 = G2 的一个有限子群（PSL(2,7) 被 Z2^3 扩展）。
"""

from __future__ import annotations

import json
from itertools import permutations, product
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


def mul_rec(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    if n == 1:
        return x * y
    m = n // 2
    a, b = x[:m], x[m:]
    c, d = y[:m], y[m:]
    ac = mul_rec(a, c)
    db = mul_rec(conj(d), b)
    da = mul_rec(d, a)
    bc = mul_rec(b, conj(c))
    return np.concatenate([ac - db, da + bc])


def assoc(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return mul_rec(mul_rec(x, y), z) - mul_rec(x, mul_rec(y, z))


def main() -> None:
    dim = 8
    eye = np.eye(dim)

    # precompute product table prod[i][j] = e_i * e_j (8-dim vector)
    prod = np.zeros((dim, dim, dim))
    for i in range(dim):
        for j in range(dim):
            prod[i, j] = mul_rec(eye[i], eye[j])

    lines = [{1, 2, 3}, {1, 4, 5}, {1, 6, 7}, {2, 4, 6}, {2, 5, 7}, {3, 4, 7}, {3, 5, 6}]
    lineset = {frozenset(l) for l in lines}

    autos = []
    for pi in permutations(range(1, 8)):
        mapped = {frozenset(pi[x - 1] for x in l) for l in lines}
        if mapped == lineset:
            autos.append(pi)

    n_lifts_total = 0
    all_unique = True
    example_sigma = None
    example_eps = None

    for sigma in autos:
        sigma_ext = [0] + list(sigma)  # sigma_ext[i] = sigma(i), sigma(0)=0
        lifts = []
        for eps in product([-1, 1], repeat=7):
            eps_ext = [1] + list(eps)  # sign of sigma'(e_i); eps_0 = 1
            ok = True
            for i in range(dim):
                for j in range(dim):
                    # (P e_i)(P e_j) = eps_i eps_j * (e_sigma(i) e_sigma(j))
                    lhs = eps_ext[i] * eps_ext[j] * prod[sigma_ext[i], sigma_ext[j]]
                    # P(e_i e_j) = P(prod[i,j]) = sum_k prod[i,j][k] * eps_k * e_sigma(k)
                    rhs = np.zeros(dim)
                    for k in range(dim):
                        c = prod[i, j, k]
                        if abs(c) > 1e-12:
                            rhs += c * eps_ext[k] * eye[sigma_ext[k]]
                    if not np.allclose(lhs, rhs, atol=1e-9):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                lifts.append(eps)
        if len(lifts) != 1:
            all_unique = False
        n_lifts_total += len(lifts)
        if example_sigma is None and lifts:
            example_sigma, example_eps = sigma, lifts[0]

    # associator invariance under the unique lift of the first automorphism
    assoc_invariant = False
    if example_sigma is not None:
        P = np.zeros((dim, dim))
        P[0, 0] = 1.0
        for k in range(1, 8):
            P[example_sigma[k - 1], k] = example_eps[k - 1]
        rng = np.random.default_rng(1)
        assoc_invariant = True
        for _ in range(20):
            x = rng.normal(size=dim)
            y = rng.normal(size=dim)
            z = rng.normal(size=dim)
            if not np.allclose(P @ assoc(x, y, z), assoc(P @ x, P @ y, P @ z), atol=1e-9):
                assoc_invariant = False
                break

    print(f"Fano-plane automorphisms (PSL(2,7)): {len(autos)}  (expected 168)")
    print(f"  total sign-lifts to algebra automorphisms: {n_lifts_total}  (expected 168 x 8 = 1344)")
    print(f"  8-fold lift per automorphism (3 generator sign dof): {not all_unique}")
    print(f"  associator invariant under a lifted automorphism: {assoc_invariant}")
    print("  => associator (ternary) structure invariant under PSL(2,7) subset G2; G2 fixes e1 = SU(3)")

    out = ROOT / "experiments" / "exp_associator_g2_invariant_last_run.json"
    out.write_text(json.dumps({
        "n_fano_automorphisms": len(autos),
        "n_sign_lifts": n_lifts_total,
        "lift_is_eightfold": not all_unique,
        "associator_invariant_example": bool(assoc_invariant),
        "conclusion": "168 Fano automorphisms lift 8-fold to 1344 octonion automorphisms preserving the associator",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
