"""Is the "-b" in the Cayley-Dickson conjugation FORCED by "norm must be real"?

Conjecture to test: 断裂 = Cayley-Dickson doubling, with the mapping
  - 断裂 (1->2)          = doubling (A -> A (+) A)
  - 观察 J (J^2 = -1)    = new imaginary unit i (i^2 = -1)
  - 自反性 (no external observer) = conjugation (a,b)* = (a*, -b)

Key question: where does the "-b" (negation) in the conjugation come from?
Claim: it is FORCED by requiring the norm N(x) = x x* to be REAL (self-conjugate),
i.e. "no external observer" => "the norm is real" => the "-b" is forced.

Verify: try conjugation with +b vs -b; only -b gives real norm.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def mul_C(x, y):
    """C = R (+) R, multiplication (a,b)(c,d) = (ac - db, da + bc)."""
    a, b = x[0], x[1]
    c, d = y[0], y[1]
    return np.array([a * c - d * b, d * a + b * c])


def conj(x, negate_second):
    """Conjugation (a,b)* = (a, sign*b)."""
    return np.array([x[0], -x[1] if negate_second else x[1]])


def norm(x, negate_second):
    return mul_C(x, conj(x, negate_second))


def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=2)

    # imaginary unit i = (0,1)
    i = np.array([0.0, 1.0])
    i2 = mul_C(i, i)

    n_neg = norm(x, negate_second=True)   # (a,b)* = (a,-b)
    n_pos = norm(x, negate_second=False)  # (a,b)* = (a,+b)

    print("Cayley-Dickson doubling R -> C, elements (a,b), mult (a,b)(c,d)=(ac-db, da+bc)")
    print(f"  imaginary unit i=(0,1):  i^2 = {i2}  (= -1, matches 观察 J: J^2=-1)")
    print()
    print(f"  random x = {np.round(x, 4)}")
    print(f"  norm with (a,b)*=(a,-b):  N(x) = {np.round(n_neg, 6)}  -> real? {abs(n_neg[1]) < 1e-12}")
    print(f"  norm with (a,b)*=(a,+b):  N(x) = {np.round(n_pos, 6)}  -> real? {abs(n_pos[1]) < 1e-12}")
    print()
    print("Conclusion:")
    print("  The '-b' in the conjugation is FORCED by requiring N(x) = x x* to be REAL.")
    print("  'norm real' = self-conjugate = 自反性 (no external observer).")
    print("  => the conjugation's negative sign comes from 自反性, NOT from 号差.")
    print("     (号差 = 观察 = J, J^2=-1, is the imaginary unit, a separate ingredient.)")

    out = ROOT / "experiments" / "exp_norm_forces_conjugation_last_run.json"
    out.write_text(json.dumps({
        "i_squared": i2.tolist(),
        "norm_neg_second": n_neg.tolist(),
        "norm_pos_second": n_pos.tolist(),
        "neg_gives_real": bool(abs(n_neg[1]) < 1e-12),
        "pos_gives_real": bool(abs(n_pos[1]) < 1e-12),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
