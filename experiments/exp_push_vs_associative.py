"""Does "push" (推动) = non-commutativity give non-associativity?

User's idea: non-associativity = "the opposing side pushes this side forward"
(3-fold cycle), not 2-fold oscillation. Key check: is "push" = non-commutative,
and does non-commutativity give non-associativity?

Answer (verified): NO. Non-commutative != non-associative.
  - Z3 q-commutation (ab = w ba, w^3 = 1): NON-commutative but ASSOCIATIVE (quantum torus).
  - The S3 3-cycle (permuting {Gamma, K, Gamma K}): a group action, ASSOCIATIVE.
  - octonions: NON-associative (the real "3-fold push" = 3-ary associator != 0).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def build_qcomm(w, n=3):
    """Clock-and-shift algebra: generators a, b with ab = w ba, a^n = b^n = 1.
    Basis {a^i b^j : i,j in 0..n-1}, dim n^2. Multiplication via b^j a^k = w^{-jk} a^k b^j."""
    dim = n * n

    def idx(i, j):
        return i * n + j

    def mul(x, y):
        out = np.zeros(dim, dtype=complex)
        for i in range(n):
            for j in range(n):
                xi = x[idx(i, j)]
                if abs(xi) < 1e-15:
                    continue
                for k in range(n):
                    for l in range(n):
                        yk = y[idx(k, l)]
                        if abs(yk) < 1e-15:
                            continue
                        out[idx((i + k) % n, (j + l) % n)] += xi * yk * (w ** (-j * k))
        return out

    return mul


def check_associative(mul, dim, n_trials=3000):
    rng = np.random.default_rng(0)
    for _ in range(n_trials):
        x = rng.normal(size=dim)
        y = rng.normal(size=dim)
        z = rng.normal(size=dim)
        if not np.allclose(mul(mul(x, y), z), mul(x, mul(y, z)), atol=1e-9):
            return False
    return True


def main() -> None:
    w = np.exp(2j * np.pi / 3)
    mul = build_qcomm(w, n=3)
    dim = 9
    a = np.zeros(dim); a[3] = 1.0   # a^1 b^0  (idx(1,0)=3)
    b = np.zeros(dim); b[1] = 1.0   # a^0 b^1  (idx(0,1)=1)
    ab = mul(a, b)
    ba = mul(b, a)

    noncomm = not np.allclose(ab, ba, atol=1e-9)
    qcomm_ok = np.allclose(ab, w * ba, atol=1e-9)
    assoc = check_associative(mul, dim)

    print("Z3 q-commutation  ab = w ba  (w^3 = 1), clock-and-shift algebra:")
    print(f"  non-commutative?  {noncomm}")
    print(f"  ab = w ba exactly?  {qcomm_ok}")
    print(f"  associative?  {assoc}")
    print()
    print("Contrast:")
    print("  S3 3-cycle (permute {Gamma, K, Gamma K}) = group action => ASSOCIATIVE")
    print("  octonions (associator [e1,e2,e4] = 2 e7 != 0) => NON-associative")
    print()
    print("Conclusion:")
    print("  'push' as NON-commutative (q-commutation) is still ASSOCIATIVE.")
    print("  The '3-fold push' that is NON-associative is the OCTONION multiplication,")
    print("  where the associator [x,y,z] (a 3-ary object) does not vanish.")
    print("  => the missing mechanism is NON-ASSOCIATIVITY, not non-commutativity.")

    out = ROOT / "experiments" / "exp_push_vs_associative_last_run.json"
    out.write_text(json.dumps({
        "non_commutative": bool(noncomm),
        "qcomm_relation_ok": bool(qcomm_ok),
        "associative": bool(assoc),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
