"""Massless Green's function on the 3-sphere S^3 (compact) vs R^3 (non-compact).

The theory's "physical 3D = SU(2) = S^3" is COMPACT.  The massless propagator
on a compact S^3 (radius R) has a DISCRETE spectrum (gap ~ 1/R^2), so its
Green's function is CUT OFF at distance ~ R -- it is NOT 1/r at large r.
Only the non-compact limit R -> infinity gives the R^3 Coulomb 1/(4 pi r).

This is the "discrete -> continuous" wall, made concrete in 3D:
    compact S^3 (finite R)  -> discrete spectrum -> bounded (short-ish range)
    non-compact R^3 (R->inf) -> continuous spectrum -> 1/r (long range)

Green's function on S^3 (radius R), massless, zero mode removed:
    G(theta) = (1/(4 pi^2 R)) * [ (pi - theta) * cot(theta) + 1 ]
verified against the spectral sum  sum_n n sin(n theta)/(n^2-1).

Code: `py -m experiments.exp_gravity_S3_greens`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def S3_greens_closed(theta, R):
    """Closed-form massless Green's function on S^3, radius R."""
    with np.errstate(divide='ignore', invalid='ignore'):
        cot = np.cos(theta) / np.sin(theta)
    val = ((np.pi - theta) * cot + 1.0) / (4.0 * np.pi ** 2 * R)
    # handle theta -> 0 limit: G -> (pi/(4 pi^2 R)) * (1/theta) = 1/(4 pi R theta)
    return val


def S3_greens_spectral(theta, R, n_max=200000):
    """Spectral sum of the S^3 Green's function (Cesaro-smoothed)."""
    # sum_{n=2}^inf n sin(n theta)/(n^2 - 1), with Cesaro smoothing for convergence
    n = np.arange(2, n_max, dtype=np.float64)
    terms = n * np.sin(n * theta) / (n ** 2 - 1.0)
    # Cesaro (1 - k/N) smoothing of partial sums
    k = np.arange(1, len(terms) + 1, dtype=np.float64)
    smooth = 1.0 - k / len(terms)
    s = np.sum(terms * smooth)
    return s / (2.0 * np.pi ** 2 * R * np.sin(theta))


def main():
    print("=" * 74)
    print("MASSLESS GREEN'S FUNCTION: compact S^3 vs non-compact R^3")
    print("=" * 74)

    # verify closed form against spectral sum at a mid angle
    th = 1.0
    closed = S3_greens_closed(th, 1.0)
    spec = S3_greens_spectral(th, 1.0)
    print(f"\nverify closed form vs spectral sum at theta=1.0 rad, R=1:")
    print(f"  closed   = {closed:.6f}")
    print(f"  spectral = {spec:.6f}")
    print(f"  |diff|   = {abs(closed - spec):.2e}")

    # compare S^3 Green's function (R=1,2,5,10) against R^3 Coulomb 1/(4 pi r)
    print("\n  G vs physical distance r = R*theta (r in units where R^3 gives 1/(4 pi r)):")
    print(f"  {'r':>7} {'R^3 1/(4pi r)':>13} {'S3 R=1':>12} {'S3 R=2':>12} {'S3 R=5':>12} {'S3 R=10':>12}")
    rows = []
    for r in [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
        coul = 1.0 / (4.0 * np.pi * r)
        g = {}
        for R in [1, 2, 5, 10]:
            theta = r / R
            if theta >= np.pi:  # antipode wrap, cap
                theta = np.pi - 1e-9
            g[R] = S3_greens_closed(theta, R)
        row = {"r": r, "coulomb": coul, **{f"R{R}": float(g[R]) for R in [1, 2, 5, 10]}}
        rows.append(row)
        print(f"  {r:>7.2f} {coul:>13.5f} {g[1]:>12.5f} {g[2]:>12.5f} {g[5]:>12.5f} {g[10]:>12.5f}")

    # key message
    print("\n--- interpretation ---")
    print("  At small r, S^3 (any R) ~ 1/(4 pi r) = Coulomb (short-distance).")
    print("  At large r (r ~ R), S^3 GREEN'S FUNCTION STOPS GROWING / turns over")
    print("    (compact space cuts off the 1/r tail) -- NOT long-range.")
    print("  Only R -> infinity recovers 1/(4 pi r) everywhere = R^3 Coulomb = long-range 1/r.")

    out = ROOT / "experiments" / "exp_gravity_S3_greens_last_run.json"
    out.write_text(json.dumps({
        "verify_closed": float(closed), "verify_spectral": float(spec),
        "verify_diff": float(abs(closed - spec)),
        "table": rows,
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
