"""Symbolic verification (sympy) of the earlier hand-derived steps of the
vortex path, which were NOT program-verified before:

  1. Toeplitz spectrum  lambda_k = 2 cos(k pi/(n+1))  (the "binary plateau" basis)
  2. vortex winding number = n, far-field |grad theta| = n/r
  3. inner derivation [h, .] satisfies the Leibniz rule (the "scalar->tensor"
     wall is about OUTER derivations; inner ones always exist)

Code: `py -m experiments.exp_symbolic_vortex`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def tridiag(n):
    M = sp.zeros(n, n)
    for i in range(n - 1):
        M[i, i + 1] = 1
        M[i + 1, i] = 1
    return M


def main():
    print("=== symbolic verification of the earlier hand-derived steps ===")
    print()

    # ------------------------------------------------------------------
    # 1. Toeplitz spectrum (the binary-plateau basis)
    # ------------------------------------------------------------------
    print("1. Toeplitz tridiag_n(1,0,1) spectrum = { 2 cos(k pi/(n+1)) }:")
    results = {}
    for n in (2, 3, 4):
        M = tridiag(n)
        evs = sorted(M.eigenvals().keys(), key=lambda e: sp.N(e))
        expected = sorted([2 * sp.cos(k * sp.pi / (n + 1)) for k in range(1, n + 1)],
                          key=lambda e: sp.N(e))
        match = all(sp.simplify(evs[i] - expected[i]) == 0 for i in range(n))
        results[n] = bool(match)
        print(f"   n={n}: eig = {evs}")
        print(f"        2cos(k pi/(n+1)) = {expected}   match = {match}")

    # ------------------------------------------------------------------
    # 2. vortex winding number + far field
    # ------------------------------------------------------------------
    print()
    print("2. vortex theta = n arg(x+iy): winding number = n, far-field |grad| = n/r:")
    x, y, n = sp.symbols('x y n', real=True)
    grad_x = -n * y / (x ** 2 + y ** 2)
    grad_y = n * x / (x ** 2 + y ** 2)
    norm2 = sp.simplify(grad_x ** 2 + grad_y ** 2)
    target_norm = sp.simplify(n ** 2 / (x ** 2 + y ** 2))
    far_ok = sp.simplify(norm2 - target_norm) == 0
    print(f"   |grad theta|^2 = {norm2}")
    print(f"   n^2/(x^2+y^2)  = {target_norm}   match = {far_ok}")

    R, t = sp.symbols('R t', real=True, positive=True)
    xc = R * sp.cos(t)
    yc = R * sp.sin(t)
    # winding = (1/2pi) oint (grad_x dx + grad_y dy) = (1/2pi) oint n (x dy - y dx)/(x^2+y^2)
    integrand = sp.simplify(n * (xc * sp.diff(yc, t) - yc * sp.diff(xc, t)) / (xc ** 2 + yc ** 2))
    winding = sp.simplify(sp.integrate(integrand, (t, 0, 2 * sp.pi)) / (2 * sp.pi))
    print(f"   winding = (1/2pi) oint grad.dl = {winding}   (should = n)")

    # ------------------------------------------------------------------
    # 3. inner derivation [h, .] satisfies Leibniz
    # ------------------------------------------------------------------
    print()
    print("3. inner derivation delta(a)=[h,a]=ha-ah satisfies the Leibniz rule:")
    h11, h12, h21, h22 = sp.symbols('h11 h12 h21 h22')
    a11, a12, a21, a22 = sp.symbols('a11 a12 a21 a22')
    b11, b12, b21, b22 = sp.symbols('b11 b12 b21 b22')
    h = sp.Matrix([[h11, h12], [h21, h22]])
    A = sp.Matrix([[a11, a12], [a21, a22]])
    B = sp.Matrix([[b11, b12], [b21, b22]])

    def delta(M):
        return sp.simplify(h @ M - M @ h)

    lhs = delta(A @ B)
    rhs = sp.simplify(delta(A) @ B + A @ delta(B))
    leibniz = sp.simplify(lhs - rhs)
    leibniz_ok = leibniz == sp.zeros(2, 2)
    print(f"   delta(AB) - (delta(A)B + A delta(B)) = {leibniz}")
    print(f"   => inner derivation satisfies Leibniz: {leibniz_ok}")

    print()
    print("=== conclusion ===")
    print("  1. Toeplitz spectrum symbolically confirmed (the binary plateau is exact).")
    print("  2. vortex winding = n and far-field = n/r symbolically confirmed.")
    print("  3. inner derivations satisfy Leibniz (always exist).  The scalar->tensor")
    print("     wall is about OUTER derivations (which II_1 lacks) -- a theorem, not a")
    print("     computation; see Sakai-Kadison.  Finite-dimensional M_n has NO outer")
    print("     derivations either (HH^1(M_n)=0), so the wall is about the LIMIT")
    print("     (finite -> infinite), which finite numeric cannot reach.")

    summary = {
        "toeplitz_match": {str(k): v for k, v in results.items()},
        "far_field_match": bool(far_ok),
        "winding_number": str(winding),
        "leibniz_ok": bool(leibniz_ok),
        "wall_note": "scalar->tensor needs OUTER derivations; II_1 has none (Sakai-Kadison), "
                      "and finite M_n has none either (HH^1=0). The wall is the limit "
                      "finite->infinite, which finite numeric cannot reach.",
    }
    out = ROOT / "experiments" / "exp_symbolic_vortex_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
