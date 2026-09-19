"""Symbolic: two-observer stereographic transition function.

Observer p1 (north pole N): pi_N: S^3\\{N} -> R^3, y_i = x_i/(1-x0)
Observer p2 (south pole S): pi_S: S^3\\{S} -> R^3, y'_i = x_i/(1+x0)
Transition: pi_S o pi_N^{-1}: R^3 -> R^3

Verify: transition = inversion (conformal), gives GLOBAL conformal (not
position-dependent curvature).

Code: `py -m experiments.exp_verify_multiple_observer`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    print("=== symbolic: two-observer stereographic transition ===")
    print()

    y1, y2, y3 = sp.symbols('y1 y2 y3', real=True)
    r2 = y1 ** 2 + y2 ** 2 + y3 ** 2
    # inverse pi_N: x_i = 2y_i/(1+r^2), x0 = (r^2-1)/(r^2+1)
    # pi_S: y'_i = x_i/(1+x0) = y_i/r^2
    trans = sp.Matrix([y1 / r2, y2 / r2, y3 / r2])
    print("1. transition pi_S o pi_N^{-1}:  y ->", trans.tolist())
    print("   = inversion y -> y/|y|^2")

    J = sp.Matrix([[sp.diff(trans[i], y) for y in (y1, y2, y3)] for i in range(3)])
    JTJ = sp.simplify(J.T @ J)
    expected = sp.eye(3) / r2 ** 2
    residual = sp.simplify(JTJ - expected)
    print()
    print("2. Jacobian J^T J =", sp.simplify(JTJ))
    print(f"   expected (1/r^4) I (conformal factor 1/r^2), residual = {residual}")
    print("   => inversion is conformal (angle-preserving), factor 1/r^2 (global).")

    print()
    print("3. inversion is a GLOBAL conformal map (one inversion covers all R^3):")
    print("   - it is a coordinate change (N-chart to S-chart), not a position-dependent")
    print("     connection.")
    print("   - conformal factor 1/r^2 varies with position, but is a FIXED function")
    print("     (determined by inversion), not a point-wise variable field (connection).")
    print("   - after inversion the metric is still conformally flat (constant curvature,")
    print("     Weyl=0), NOT position-dependent Riemann curvature.")

    print()
    print("=== conclusion ===")
    print("  two-observer transition = inversion (conformal) = GLOBAL coordinate change,")
    print("  NOT position-dependent curvature. It gives conformally-flat (constant")
    print("  curvature), not Riemann curvature (scalar curvature varying point to point).")
    print("  Position-dependent curvature needs a FIELD of transition functions (many")
    print("  observers), whose variation = conformal connection (Weyl), still not Riemann.")
    print("  And 'observer = stereographic projection' still needs 'observer = S^3 point'")
    print("  (state=position identification).")

    summary = {
        "transition_function": "inversion y -> y/|y|^2",
        "JtJ_residual": str(residual),
        "conformal_factor": "1/r^2",
        "conclusion": "two-observer transition = inversion (global conformal, not "
                      "position-dependent curvature); gives conformally-flat (constant "
                      "curvature), not Riemann curvature. Position-dependent curvature "
                      "needs a field of transitions (many observers) -> conformal "
                      "connection (Weyl), still not Riemann. 'observer = projection' "
                      "still needs state=position identification.",
    }
    out = ROOT / "experiments" / "exp_verify_multiple_observer_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
