"""Candidate (soft-mode): does matter push the modulus mass m^2(rho) toward zero?

User's idea: "equal-moduli relaxation" = the modulus field r has an effective mass
m^2 that DECREASES with matter density rho.  At a critical rho_c, m^2 -> 0 => the
modulus becomes MASSLESS => long-range gravity (a collective effect of matter
condensation, not a pre-existing massless geometry).

This is a standard soft-mode / quantum-critical-point mechanism.  It does NOT
hand-place a coupling: matter-modulus coupling is built-in (fermion hopping ~ r).

HONEST technical point: the modulus mass must be defined for an INHOMOGENEOUS
modulus perturbation (staggered), because E_gs is LINEAR in a UNIFORM rescaling
(spectrum ~ r), so the uniform Hessian is identically 0.

Compute: m^2(E_F) = d^2 E_gs / d eps^2  for a staggered modulus perturbation
D -> D + eps * M,  M_ij = (-1)^{x+y} D_ij,  where E_gs(E_F) = sum_{E_n < E_F} E_n.
Scan the Fermi energy E_F (proxy for matter density rho).  If m^2(E_F) is negative
and grows more negative with E_F, then adding the (positive) trace-action mass
could push the total m^2 to zero at some rho_c => critical point exists.

Code: `py -m experiments.exp_bridge_B_softmode`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from itertools import product

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def torus2D(L, flux=True):
    N = L * L

    def idx(x, y):
        return (y % L) * L + (x % L)

    D = np.zeros((N, N))
    for x, y in product(range(L), repeat=2):
        i = idx(x, y)
        j = idx(x + 1, y); w = (-1) ** y if flux else 1.0; D[i, j] = w; D[j, i] = w
        j = idx(x, y + 1); w = 1.0; D[i, j] = w; D[j, i] = w
    return D


def staggered_mask(D, L):
    """M_ij = (-1)^{x+y} D_ij (staggered modulation of the hopping)."""
    N = L * L

    def idx(x, y):
        return (y % L) * L + (x % L)

    M = np.zeros((N, N))
    for x, y in product(range(L), repeat=2):
        i = idx(x, y)
        j = idx(x + 1, y)
        if D[i, j] != 0:
            M[i, j] = (-1) ** (x + y) * D[i, j]
            M[j, i] = M[i, j]
        j = idx(x, y + 1)
        if D[i, j] != 0:
            M[i, j] = (-1) ** (x + y) * D[i, j]
            M[j, i] = M[i, j]
    return M


def Egs(D, EF):
    """Ground-state (Fermi-sea) energy at Fermi energy EF: sum of E_n for E_n < EF."""
    eigs = np.linalg.eigvalsh(D)
    return float(eigs[eigs < EF].sum())


def main():
    L = 32
    D = torus2D(L, flux=True)
    M = staggered_mask(D, L)
    eps = 0.05

    # eigenvalues of unperturbed D for reference (Dirac point at E=0)
    e0 = np.linalg.eigvalsh(D)
    print("=== soft mode: m^2(E_F) = d^2 Egs/d eps^2 for staggered modulus ===")
    print(f"  L={L}, eps={eps}, staggered perturbation D -> D + eps*M")
    print(f"  spectrum of D: min={e0.min():.3f}, max={e0.max():.3f} (Dirac point E=0)")

    # scan Fermi energy E_F over the band
    EFs = np.linspace(-2.0, 0.0, 9)  # from deep valence to half-filling (E=0)
    print(f"\n  {'E_F':>6} {'m^2(E_F)':>14}")
    results = []
    for EF in EFs:
        Ep = Egs(D + eps * M, EF)
        E0 = Egs(D, EF)
        Em = Egs(D - eps * M, EF)
        m2 = (Ep - 2 * E0 + Em) / (eps ** 2)
        results.append({"EF": float(EF), "m2": float(m2)})
        print(f"  {EF:>6.2f} {m2:>14.6e}")

    # also slightly above half filling (hole doping) E_F > 0
    for EF in [0.2, 0.5]:
        Ep = Egs(D + eps * M, EF)
        E0 = Egs(D, EF)
        Em = Egs(D - eps * M, EF)
        m2 = (Ep - 2 * E0 + Em) / (eps ** 2)
        results.append({"EF": float(EF), "m2": float(m2)})
        print(f"  {EF:>6.2f} {m2:>14.6e}")

    # read off sign and trend
    m2s = np.array([r["m2"] for r in results])
    print(f"\n  => sign of m^2(E_F): {'all negative' if (m2s < 0).all() else 'all positive' if (m2s > 0).all() else 'mixed'}")
    print(f"  => if m^2 < 0 (fermion back-reaction softens the modulus), then a")
    print(f"     critical rho_c exists where total m^2 (trace + fermi) -> 0.")

    summary = {
        "L": L, "eps": eps,
        "scan": results,
        "conclusion": "m^2(E_F) for staggered modulus perturbation. If negative and "
                      "growing more negative with |E_F|, the fermion back-reaction "
                      "softens the modulus; adding the (positive) trace-action mass "
                      "could give a critical density where m^2 -> 0 (soft mode / massless "
                      "geometry). If positive, no softening: the modulus stays massive.",
    }
    out = ROOT / "experiments" / "exp_bridge_B_softmode_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
