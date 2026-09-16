"""Cl(0,3) vs octonions: does "3 anti-commuting generators" give O, or Cl(3)?

The theory's 3D pi-flux has 3 anti-commuting magnetic translations (T_x T_y T_z).
Question: do 3 anti-commuting generators give the octonions O (-> SU(3) via G2),
or the Clifford algebra Cl(0,3) = H⊕H (associative, WITH zero divisors)?

Answer (verified below): Cl(0,3) is associative AND has zero divisors, e.g.
    (1 + e123)(1 - e123) = 0,
so it is NOT a division algebra -> NOT on the Hurwitz ladder (R,C,H,O).
The octonions O are non-associative but division (no zero divisors).
=> "3 anti-commuting" gives Cl(3), NOT O. The climb H -> O needs the
non-associative Cayley-Dickson step, which anti-commutation does NOT provide.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def cliff_sign(a: int, b: int) -> int:
    """e_A e_B = sign * e_{A xor B}, for Cl(0,3) (e_i^2 = -1).

    sign = (-1)^{inv(a,b) + popcount(a&b)}, inv = #{(i,j): i in A, j in B, i>j}.
    """
    inv = 0
    for i in range(3):
        if (a >> i) & 1:
            for j in range(i):
                if (b >> j) & 1:
                    inv += 1
    common = bin(a & b).count("1")
    return 1 if (inv + common) % 2 == 0 else -1


def build_table(n: int = 3):
    dim = 1 << n
    sign = np.zeros((dim, dim))
    idx = np.zeros((dim, dim), dtype=int)
    for a in range(dim):
        for b in range(dim):
            s = cliff_sign(a, b)
            sign[a, b] = s
            idx[a, b] = a ^ b
    return sign, idx


def mul(x: np.ndarray, y: np.ndarray, sign, idx) -> np.ndarray:
    contrib = (x[:, None] * y[None, :]) * sign
    out = np.zeros_like(x)
    np.add.at(out, idx, contrib)
    return out


def main() -> None:
    dim = 8
    sign, idx = build_table(3)
    rng = np.random.default_rng(0)

    # associativity (Clifford algebras are associative)
    assoc_ok = True
    for _ in range(5000):
        x = rng.normal(size=dim)
        y = rng.normal(size=dim)
        z = rng.normal(size=dim)
        if not np.allclose(mul(mul(x, y, sign, idx), z, sign, idx),
                           mul(x, mul(y, z, sign, idx), sign, idx), atol=1e-9):
            assoc_ok = False
            break

    # explicit zero divisor: (1 + e123)(1 - e123)
    e0 = np.zeros(dim); e0[0] = 1.0
    e123 = np.zeros(dim); e123[7] = 1.0
    x = e0 + e123
    y = e0 - e123
    xy = mul(x, y, sign, idx)

    # also check norm multiplicativity (split algebra H⊕H: NOT normed)
    worst_gap = 0.0
    for _ in range(5000):
        a = rng.normal(size=dim)
        b = rng.normal(size=dim)
        gap = abs(float(np.dot(mul(a, b, sign, idx), mul(a, b, sign, idx))) - float(np.dot(a, a)) * float(np.dot(b, b)))
        worst_gap = max(worst_gap, gap)

    print("Cl(0,3)  (3 anti-commuting generators, e_i^2 = -1):")
    print(f"  dim = {dim}")
    print(f"  associative = {assoc_ok}")
    print(f"  (1 + e123)(1 - e123) = {np.round(xy, 10)}   -> zero divisor? {bool(np.allclose(xy, np.zeros(dim)))}")
    print(f"  normed (N(xy)=N(x)N(y))? worst_gap = {worst_gap:.3e}   ({'NO' if worst_gap > 1e-6 else 'yes'})")
    print()
    print("Contrast (from exp_division_ladder):")
    print("  O  (octonions):  associative = False,  zero divisor = False  (division algebra)")
    print("  Cl(0,3) = H⊕H:   associative = True,   zero divisor = True   (split, NOT division)")
    print()
    print("=> 3 anti-commuting generators give Cl(3)=H⊕H, NOT the octonions.")
    print("   The octonion climb (H->O) needs NON-associative Cayley-Dickson, which")
    print("   anti-commutation does not provide.")

    out = ROOT / "experiments" / "exp_clifford_vs_octonion_last_run.json"
    out.write_text(json.dumps({
        "Cl03_dim": dim,
        "Cl03_associative": bool(assoc_ok),
        "Cl03_zero_divisor": bool(np.allclose(xy, np.zeros(dim))),
        "Cl03_worst_norm_gap": worst_gap,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
