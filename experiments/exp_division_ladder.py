"""Division-algebra ladder: Cayley-Dickson R -> C -> H -> O -> S, and where it breaks.

Anchors the Hurwitz route for algebra A (为什么 C⊕H⊕M₃): the normed division
algebras are exactly R, C, H, O (dim 1,2,4,8). This script VERIFIES numerically where
each property breaks, so "the ladder stops at O" is computation, not hand-waving.

Checks per algebra:
  - commutative:        xy = yx
  - associative:        (xy)z = x(yz)     (FAILS at O -> O is NOT a Clifford algebra)
  - normed:             N(xy) = N(x)N(y)  (composition algebras R,C,H,O; FAILS at S)
  - zero divisors:      xy = 0 with x,y != 0  (appear at S)

Multiplication uses a precomputed table (e_i e_j = S[i,j] e_{K[i,j]}), vectorized.
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


def mul_rec(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cayley-Dickson multiplication (recursive), used only to build the table."""
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


def build_tables(dim: int):
    """S[i,j], K[i,j] with e_i e_j = S[i,j] * e_{K[i,j]} (S=0 => product is zero)."""
    S = np.zeros((dim, dim))
    K = np.zeros((dim, dim), dtype=int)
    for i in range(dim):
        ei = np.eye(dim)[i]
        for j in range(dim):
            prod = mul_rec(ei, np.eye(dim)[j])
            nz = np.nonzero(np.abs(prod) > 1e-10)[0]
            if len(nz) == 0:
                S[i, j] = 0.0
                K[i, j] = 0
            else:
                k = int(nz[0])
                S[i, j] = 1.0 if prod[k] > 0 else -1.0
                K[i, j] = k
    return S, K


def mul_fast(x: np.ndarray, y: np.ndarray, S: np.ndarray, K: np.ndarray, dim: int) -> np.ndarray:
    contrib = (x[:, None] * y[None, :]) * S
    out = np.zeros(dim)
    np.add.at(out, K, contrib)
    return out


def norm2(x: np.ndarray) -> float:
    return float(np.dot(x, x))


def main() -> None:
    rng = np.random.default_rng(0)
    names = ["R", "C", "H", "O", "S"]
    dims = [1, 2, 4, 8, 16]
    n_rand = 4000
    results = {}

    for name, dim in zip(names, dims):
        S, K = build_tables(dim)
        comm_ok = True
        assoc_ok = True
        norm_ok = True
        worst_gap = 0.0
        for _ in range(n_rand):
            x = rng.normal(size=dim)
            y = rng.normal(size=dim)
            z = rng.normal(size=dim)
            xy = mul_fast(x, y, S, K, dim)
            if not np.allclose(xy, mul_fast(y, x, S, K, dim), atol=1e-9):
                comm_ok = False
            if not np.allclose(mul_fast(xy, z, S, K, dim), mul_fast(x, mul_fast(y, z, S, K, dim), S, K, dim), atol=1e-9):
                assoc_ok = False
            gap = abs(norm2(xy) - norm2(x) * norm2(y))
            worst_gap = max(worst_gap, gap)
            if gap > 1e-9:
                norm_ok = False

        zd = None
        if dim == 16:
            for _ in range(300000):
                a = np.zeros(dim)
                b = np.zeros(dim)
                a[rng.integers(0, dim, size=3)] = rng.integers(-1, 2, size=3)
                b[rng.integers(0, dim, size=3)] = rng.integers(-1, 2, size=3)
                if norm2(a) > 0 and norm2(b) > 0 and np.allclose(mul_fast(a, b, S, K, dim), np.zeros(dim), atol=1e-10):
                    zd = [a.tolist(), b.tolist()]
                    break

        results[name] = {
            "dim": dim,
            "commutative": bool(comm_ok),
            "associative": bool(assoc_ok),
            "normed": bool(norm_ok),
            "worst_norm_gap": worst_gap,
            "zero_divisor_found": zd is not None,
        }
        print(f"{name:2s} dim={dim:2d}  commutative={comm_ok}  associative={assoc_ok}  "
              f"normed={norm_ok}  worst_norm_gap={worst_gap:.3e}  zero_divisor={zd is not None}")

    # correct counterexample: e1,e2,e4 with e4 = octonion's NEW unit (l).
    # (e1,e2,e3=e1e2) would sit in the H subalgebra (associative) -> NOT a counterexample.
    e1, e2, e4 = np.eye(8)[1], np.eye(8)[2], np.eye(8)[4]
    lhs = mul_rec(mul_rec(e1, e2), e4)
    rhs = mul_rec(e1, mul_rec(e2, e4))
    print("\nOctonion associativity counterexample  (e1 e2) e4  vs  e1 (e2 e4):")
    print(f"  (e1 e2) e4 -> unit e{int(np.argmax(np.abs(lhs)))}   sign {lhs[int(np.argmax(np.abs(lhs)))]:+.0f}")
    print(f"  e1 (e2 e4) -> unit e{int(np.argmax(np.abs(rhs)))}   sign {rhs[int(np.argmax(np.abs(rhs)))]:+.0f}")
    print(f"  equal? {bool(np.allclose(lhs, rhs))}   (False => O non-associative, NOT a Clifford algebra)")

    out = ROOT / "experiments" / "exp_division_ladder_last_run.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
