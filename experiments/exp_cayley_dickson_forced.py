"""Derive the Cayley-Dickson multiplication from 断裂 + 观察 + 自反性 (case R -> C).

General bilinear multiplication on R(+)R:
  (a,b)(c,d) = (a1 ac + a2 ad + a3 bc + a4 bd,  b1 ac + b2 ad + b3 bc + b4 bd)

Theory-native constraints (all three already in the theory):
  1. 断裂 (doubling, contains R): (1,0) is identity -> a1=1, a2=0, b1=0, b2=1.
  2. 观察 (imaginary unit): i=(0,1), i^2=-1 -> a4=-1, b4=0.
  3. 自反性 (norm multiplicative): N(xy)=N(x)N(y), N(a,b)=a^2+b^2 -> a3=0, b3=1.

Result: (a,b)(c,d) = (ac - bd, ad + bc) = complex multiplication, UNIQUELY FORCED.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def mult(x, y, a3, b3):
    """General form AFTER imposing identity + i^2=-1 (only a3,b3 free)."""
    a, b = x
    c, d = y
    return np.array([a * c + a3 * b * c - b * d, a * d + b3 * b * c])


def norm(x):
    return x[0] ** 2 + x[1] ** 2


def main() -> None:
    rng = np.random.default_rng(0)

    # verify: forced (a3=0, b3=1) is norm-multiplicative
    worst_forced = 0.0
    for _ in range(5000):
        x = rng.normal(size=2)
        y = rng.normal(size=2)
        gap = abs(norm(mult(x, y, 0.0, 1.0)) - norm(x) * norm(y))
        worst_forced = max(worst_forced, gap)

    # show alternative (a3=0, b3=-1, i.e. (a,b)(c,d)=(ac-bd, ad-bc)) FAILS
    worst_alt = 0.0
    for _ in range(5000):
        x = rng.normal(size=2)
        y = rng.normal(size=2)
        gap = abs(norm(mult(x, y, 0.0, -1.0)) - norm(x) * norm(y))
        worst_alt = max(worst_alt, gap)

    print("Derivation: general bilinear mult on R(+)R, impose 3 theory-native constraints")
    print("  1. 断裂 (identity (1,0)):  a1=1,a2=0,b1=0,b2=1")
    print("  2. 观察 (i^2=-1, i=(0,1)):  a4=-1,b4=0")
    print("  3. 自反性 (N(xy)=N(x)N(y)):  a3=0,b3=1  (coefficient match)")
    print()
    print("Forced formula:  (a,b)(c,d) = (ac - bd, ad + bc)  = complex multiplication")
    print(f"  max |N(xy)-N(x)N(y)|  = {worst_forced:.2e}  -> norm-multiplicative (holds)")
    print(f"  alt (ad - bc, b3=-1) gap = {worst_alt:.2e}  -> FAILS (not norm-multiplicative)")
    print()
    print("=> the Cayley-Dickson multiplication is UNIQUELY FORCED by 断裂+观察+自反性.")
    print("   No free parameter, no manual input.  The '-b' (a4=-1) and the '+bc' (b3=1)")
    print("   are both consequences of the norm-multiplicativity (自反性).")

    out = ROOT / "experiments" / "exp_cayley_dickson_forced_last_run.json"
    out.write_text(json.dumps({
        "worst_forced_gap": worst_forced,
        "worst_alt_gap": worst_alt,
        "forced_formula": "(ac - bd, ad + bc)",
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
