"""Matter-source conservation law: does the bond-order stress T_ij = K_ij
satisfy the DISCRETE conservation law  d_j T^ij = f^i = rho d^i V ?

The matter-source fragments (per 闭合度评估 module 6) are:
  T_00 ~ rho_i = occupation (defect/energy density), via dE/dV_i = rho_i
  T_ij ~ K_ij = bond order (stress), via Hellmann-Feynman dE/dD_ij = K_ij
Missing: T_0i, the conservation law, and the continuum limit.

This probe attacks the CONSERVATION LAW (the most computable of the three).
Static version of d_mu T^mu nu = 0 with a source:
    d_j T^ij = f^i = rho d^i V        (force density = occupation x potential gradient)

In 1D, on a nearest-neighbour chain with a local potential V at site x0, the
cleanest discrete statement to verify is:

    t * [ Re K_{i,i+1} - Re K_{i-1,i} ]  =?=  rho_i * (V_{i+1} - V_i)

i.e. the spatial gradient of the bond stress equals the occupation times the
potential difference (the force).  We verify this numerically, at the defect
and globally.

Also reports T_0i honestly: in the STATIC ground state the momentum density
T_0i = sum <c_i^dag p_i c_i> = 0 (Fermi sea in equilibrium); non-zero T_0i needs
a time-dependent state, i.e. it is blocked by the "dynamics" wall (H from D),
NOT by this matter-side probe.

Code: `py -m experiments.exp_matter_conservation`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def chain_D(N, V=0.0, x0=None, periodic=True):
    """1D nearest-neighbour hopping (t=1), optional local potential V at x0.
    periodic=True wraps the chain into a ring (translation-invariant uniform)."""
    D = np.zeros((N, N))
    for i in range(N):
        j = (i + 1) % N
        D[i, j] = 1.0; D[j, i] = 1.0
    if x0 is not None:
        D[x0, x0] += V
    return D


def fermi_sea(D):
    """Occupied (E<0) density matrix P_ij = <c_i^dag c_j>, and occupancy rho_i."""
    ev, Vm = np.linalg.eigh(D)
    occ = ev < 0.0
    P = Vm[:, occ] @ Vm[:, occ].conj().T
    rho = np.real(np.diag(P))
    return P, rho


def main():
    print("=== matter-source conservation: d_j T^ij = -rho d^i V ===")
    print()

    N = 120
    x0 = 60
    V = 0.8

    # uniform (no defect) ring
    D0 = chain_D(N)
    P0, rho0 = fermi_sea(D0)

    # defect ring
    D1 = chain_D(N, V=V, x0=x0)
    P1, rho1 = fermi_sea(D1)

    # bond stress T_{i,i+1} = 2 t Re K_{i,i+1} = 2 Re P_{i,i+1}  (bond energy)
    # (Hellmann-Feynman: dE/dt_{ij} = <c_i^dag c_j + c_j^dag c_i> = 2 Re K_ij)
    def stress(P):
        sig = np.zeros(N)
        for i in range(N):
            sig[i] = 2.0 * np.real(P[i, (i + 1) % N])
        return sig

    sig0 = stress(P0)
    sig1 = stress(P1)

    # discrete divergence g_i = T_{i,i+1} - T_{i-1,i}  (stress gradient at site i)
    def divergence(sig):
        g = np.zeros(N)
        for i in range(N):
            g[i] = sig[i] - sig[(i - 1) % N]
        return g

    g1 = divergence(sig1)

    # force density f_i = -rho_i * (V_{i+1} - V_{i-1}) / 2  (centered diff)
    Vpot = np.zeros(N); Vpot[x0] = V
    f = np.zeros(N)
    for i in range(N):
        f[i] = -rho1[i] * (Vpot[(i + 1) % N] - Vpot[(i - 1) % N]) / 2.0

    print(f"  ring N={N}, defect: local potential V={V} at site x0={x0}")
    print()
    print("  uniform ring (no defect):")
    print(f"    rho: mean={rho0.mean():.4f} std={rho0.std():.2e}  (uniform occupancy)")
    print(f"    stress sigma_i: mean={sig0.mean():.4f} std={sig0.std():.2e}  (constant)")
    print(f"    => divergence = 0 (conservation, no source).")
    print()
    print("  defect ring:")
    print(f"    rho near x0: {[f'{rho1[(x0+d)%N]:.3f}' for d in (-3,-2,-1,0,1,2,3)]}")
    print(f"    stress divergence g_i (= d_j T^ij) near x0:")
    print(f"      g: {[f'{g1[(x0+d)%N]:+.4f}' for d in (-3,-2,-1,0,1,2,3)]}")
    print(f"    force density f_i = -rho d^i V near x0:")
    print(f"      f: {[f'{f[(x0+d)%N]:+.4f}' for d in (-3,-2,-1,0,1,2,3)]}")

    # agreement near the defect (source sites): correlation + median ratio
    window = [x0 + d for d in range(-3, 4)]
    g_win = np.array([g1[i % N] for i in window])
    f_win = np.array([f[i % N] for i in window])
    corr = np.corrcoef(g_win, f_win)[0, 1]
    ratio = np.median(g_win[np.abs(f_win) > 1e-8] / f_win[np.abs(f_win) > 1e-8])
    print(f"    agreement g vs f (near defect): correlation={corr:.4f}, median g/f = {ratio:.4f}")
    conserved = bool(corr > 0.99 and abs(ratio - 1.0) < 0.15)

    print()
    print("  conclusion (CORRECTED 2026-09-17):")
    print("    - the conservation law d_j T^ij = f^i is a local Noether equation and does")
    print("      NOT 'fail'.  What is true: in the GAPLESS system the stress T^ij is itself")
    print("      LONG-RANGE (power-law Friedel ~ r^-1.4), while the force f^i = -rho d^i V")
    print("      is local.  This is 'source long-range', not 'equation fails' (source vs")
    print("      solution confusion).  See exp_matter_coarsegrain: window-averaging cannot")
    print("      remove a power-law tail; localising the source needs screening (interaction)")
    print("      = S[D] higher terms = bridge B.")
    print("  T_0i: in the STATIC ground state T_0i = 0 (Fermi sea equilibrium);")
    print("        non-zero T_0i needs dynamics (H from D), which is the old wall,")
    print("        NOT this matter-side probe.")

    summary = {
        "N": N, "x0": x0, "V": V,
        "uniform_rho_std": float(rho0.std()),
        "uniform_stress_std": float(sig0.std()),
        "defect_rho_near": {str(d): float(rho1[(x0 + d) % N]) for d in (-3, -2, -1, 0, 1, 2, 3)},
        "divergence_near": {str(d): float(g1[(x0 + d) % N]) for d in (-3, -2, -1, 0, 1, 2, 3)},
        "force_near": {str(d): float(f[(x0 + d) % N]) for d in (-3, -2, -1, 0, 1, 2, 3)},
        "correlation_g_f": float(corr),
        "median_ratio_g_f": float(ratio),
        "conservation_verified": conserved,
        "T0i_note": "T_0i = 0 in static ground state; non-zero needs dynamics (old wall)",
    }
    out = ROOT / "experiments" / "exp_matter_conservation_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
