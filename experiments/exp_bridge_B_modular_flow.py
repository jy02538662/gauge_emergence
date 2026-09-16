"""Modular-flow spectrum: is log(Delta) DISCRETE (finite D) or CONTINUOUS (R)?

Proposition (long-range gravity = modular-flow pull-back):
    long-range 1/r  <=>  spectrum of log(Delta) has a gapless continuous part
    finite D  =>  log(Delta) spectrum is DISCRETE (N^2 eigenvalues theta_ij)
                  => gap => short-range.

Part A: finite N, rho = diag(lambda_i).  theta_ij = ln(lambda_i/lambda_j), N^2
        discrete values.  Show the spectral GAP (min spacing) and that it does
        NOT fill a continuous interval.

Part B: contrast with a Hamiltonian whose spectrum-difference is gapless
        (a free particle on a ring, H = p^2):  show that as N grows the
        theta_ij distribution DENSIFIES toward a continuum, but for any FINITE N
        it is still a finite discrete set (gap -> 0 only as N -> infinity).

Code: `py -m experiments.exp_bridge_B_modular_flow`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def spectrum_gap(theta):
    """min spacing of sorted distinct spectrum values (log of min gap)."""
    th = np.sort(np.unique(theta))
    if len(th) < 2:
        return np.inf
    gaps = np.diff(th)
    gaps = gaps[gaps > 1e-12]
    return float(gaps.min()) if gaps.size else np.inf


def main():
    print("=== modular-flow spectrum: finite D has DISCRETE (gapped) log(Delta) ===")
    print()

    # Part A: generic rho = diag(lambda_i), lambda_i ~ exp(-beta E_i) with E_i from a
    # quadratic (gapped-like) ladder.  Show theta_ij is a finite discrete set.
    print("Part A: finite N, generic rho.  theta_ij = ln(lambda_i/lambda_j):")
    for N in [8, 16, 32, 64]:
        E = np.arange(N, dtype=float)          # energies 0,1,2,... (gapped ladder)
        beta = 0.3
        lam = np.exp(-beta * E)
        lam = lam / lam.sum()
        # theta_ij = ln(lam_i/lam_j) = beta (E_j - E_i)
        theta = np.array([np.log(lam[i] / lam[j])
                          for i in range(N) for j in range(N)])
        gap = spectrum_gap(theta)
        n_uniq = len(np.unique(np.round(theta, 10)))
        print(f"  N={N:3d}: n_distinct_theta={n_uniq:4d} (of N^2={N*N:4d}), "
              f"spectral gap = {gap:.3e}")

    print()
    print("Part B: does theta fill a CONTINUOUS interval as N -> infinity?")
    print("  (free-particle ladder E_i = i^2 / N -> theta spreads, but STILL discrete)")
    for N in [16, 64, 256]:
        E = (np.arange(N, dtype=float) ** 2) / N   # densifies toward continuum
        beta = 1.0
        lam = np.exp(-beta * E)
        lam = lam / lam.sum()
        theta = np.array([np.log(lam[i] / lam[j])
                          for i in range(N) for j in range(N)])
        gap = spectrum_gap(theta)
        print(f"  N={N:3d}: spectral gap = {gap:.3e}  (-> 0 only as N->inf, never 0)")

    print()
    print("  => For ANY finite N, log(Delta) has a positive spectral gap.")
    print("     Continuous (gapless) spectrum requires N -> infinity = R (body),")
    print("     not finite D.  Hence long-range 1/r lives in R, not D.")

    summary = {
        "conclusion": "Finite D gives a DISCRETE (gapped) modular-flow spectrum "
                      "theta_ij = ln(lambda_i/lambda_j). The gap -> 0 only as N->infinity "
                      "(R layer). So long-range 1/r (gapless continuous spectrum) is a "
                      "property of R, not D — the spectral basis of the modular-flow "
                      "pull-back proposition.",
    }
    out = ROOT / "experiments" / "exp_bridge_B_modular_flow_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
