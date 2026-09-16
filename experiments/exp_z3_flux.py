"""Z3 flux: does the trace action select flux 2π/3 (Z3), or only π (Z2)?

Build an LxL square lattice with uniform flux Φ per plaquette (Landau gauge),
compute Tr(D^2), Tr(D^4), Tr(D^6) as functions of Φ, find their minima.

Question: is 2π/3 (Z3) a natural optimum of any trace term, analogous to how
Tr(D^4) is minimized at Φ=π (Z2, the structure-selection-gate result)?
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def build_D(L: int, phi: float) -> np.ndarray:
    """LxL square lattice, uniform flux phi per plaquette (Landau gauge, open BC).

    horizontal (x,y)->(x+1,y): phase 1 ;  vertical (x,y)->(x,y+1): phase e^{i phi x}.
    """
    n = L * L
    D = np.zeros((n, n), complex)
    for x in range(L):
        for y in range(L):
            i = x + L * y
            if x + 1 < L:
                j = (x + 1) + L * y
                D[i, j] = 1.0
                D[j, i] = 1.0
            if y + 1 < L:
                j = x + L * (y + 1)
                p = np.exp(1j * phi * x)
                D[i, j] = p
                D[j, i] = np.conj(p)
    return D


def trace_power(D: np.ndarray, k: int) -> float:
    return float(np.real(np.trace(np.linalg.matrix_power(D, k))))


def main() -> None:
    L = 8
    phis = np.linspace(0.0, 2.0 * np.pi, 241)
    tr2, tr4, tr6 = [], [], []
    for phi in phis:
        D = build_D(L, phi)
        tr2.append(trace_power(D, 2))
        tr4.append(trace_power(D, 4))
        tr6.append(trace_power(D, 6))
    tr2 = np.asarray(tr2)
    tr4 = np.asarray(tr4)
    tr6 = np.asarray(tr6)

    def report(name, arr):
        imin = int(np.argmin(arr))
        imax = int(np.argmax(arr))
        print(f"{name}:  min at phi={phis[imin]:.3f} ({phis[imin]/np.pi:.3f} pi)  value={arr[imin]:.2f}   "
              f"| max at phi={phis[imax]:.3f} ({phis[imax]/np.pi:.3f} pi)  value={arr[imax]:.2f}")

    print(f"L={L} square lattice, uniform flux phi per plaquette. Trace minima:")
    report("Tr(D2)", tr2)
    report("Tr(D4)", tr4)
    report("Tr(D6)", tr6)

    # values at key fluxes
    print("\nvalues at key fluxes (phi = 0, pi/3, 2pi/3, pi):")
    for name, arr in [("Tr(D2)", tr2), ("Tr(D4)", tr4), ("Tr(D6)", tr6)]:
        vals = [float(arr[np.argmin(abs(phis - t))]) for t in [0, np.pi/3, 2*np.pi/3, np.pi]]
        print(f"  {name:8s}  phi=0: {vals[0]:8.2f}  phi=pi/3: {vals[1]:8.2f}  phi=2pi/3: {vals[2]:8.2f}  phi=pi: {vals[3]:8.2f}")

    out = ROOT / "experiments" / "exp_z3_flux_last_run.json"
    out.write_text(json.dumps({
        "L": L,
        "TrD4_min_phi": float(phis[int(np.argmin(tr4))]),
        "TrD6_min_phi": float(phis[int(np.argmin(tr6))]),
        "TrD6_at_2pi3": float(tr6[np.argmin(abs(phis - 2*np.pi/3))]),
        "TrD6_at_pi": float(tr6[np.argmin(abs(phis - np.pi))]),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
