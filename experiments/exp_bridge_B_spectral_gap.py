"""Spectral gap <-> correlation decay (1D chain) — CORRECTED.

Previous bug: computed G(r) from eigh's REAL eigenvectors, but the tight-binding
chain has k / -k degeneracy, so eigh returns cos/sin combinations and G(r)
collapsed to a delta.  Fix: for the gapless free fermion, compute G(r) directly
in momentum space (sum over occupied k of e^{ikr}); for the gapped staggered-mass
chain, diagonalize and sum over occupied Bloch states (no degeneracy issue there
because the mass breaks k -> -k into a 2-site cell).

Physics to show (standard fact, spectral core of the modular-flow proposition):
  - gapless H  =>  G(r) ~ sin(pi r/2)/(pi r),  C(r)=|G(r)|^2 ~ 1/r^2  (POWER LAW, long range)
  - gapped  H  =>  C(r) ~ exp(-r/xi)  (exponential, short range)

Code: `py -m experiments.exp_bridge_B_spectral_gap`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def corr_gapless(L):
    """Free fermion chain, half filling.  G(r) = (1/L) sum_{k occ} e^{ikr}, in momentum space."""
    G = np.zeros(L, dtype=complex)
    for m in range(L):
        k = 2 * np.pi * m / L
        E = -2.0 * np.cos(k)          # tight-binding E(k)
        if E < 0.0:                   # occupied at half filling (Fermi at E=0)
            G += np.exp(1j * k * np.arange(L)) / L
    C = np.abs(G) ** 2                # density-density = |G|^2 (Wick, free fermions)
    return C, G


def corr_gapped(L, mass):
    """Staggered-mass chain: H = -t sum c^dag c + mass * (-1)^x n_x.  Diagonalize,
    occupy E<0, compute G(r) = <c_0^dag c_r>."""
    H = np.zeros((L, L))
    for x in range(L):
        H[x, (x + 1) % L] = -1.0
        H[(x + 1) % L, x] = -1.0
    for x in range(L):
        H[x, x] = mass * ((-1) ** x)
    eigs, vecs = np.linalg.eigh(H)
    occ = eigs < 0.0
    # G(r) = sum_{occ} conj(v_0) v_r  (averaged over starting site for trans-invariance
    # is broken to a 2-site cell; average over the 2 sublattices)
    G = np.zeros(L, dtype=complex)
    for a in np.where(occ)[0]:
        v = vecs[:, a]
        for x0 in range(2):           # average over the 2-site cell
            for r in range(L):
                G[r] += np.conj(v[x0]) * v[(x0 + r) % L] / 2.0
    C = np.abs(G) ** 2
    return C, G


def main():
    L = 400
    print("=== spectral gap <-> correlation decay (1D chain, CORRECTED) ===")
    print(f"  L={L}")

    C_g, G_g = corr_gapless(L)
    C_m, G_m = corr_gapped(L, mass=0.3)

    rs = [1, 2, 3, 4, 5, 8, 11, 16, 32, 64, 128]
    print("\n  r     gapless C(r)      gapped C(r)")
    for r in rs:
        print(f"  {r:4d}  {C_g[r]:14.4e}  {C_m[r]:14.4e}")

    # fit: gapless C(r) ~ r^-alpha on ODD r (even r = 0 by the sin(pi r/2) node)
    odd = np.array([r for r in rs if r % 2 == 1])
    cg_odd = C_g[odd]
    if (cg_odd > 1e-15).all():
        alpha = -np.polyfit(np.log(odd), np.log(cg_odd), 1)[0]
        print(f"\n  gapless: C(r) ~ r^-alpha on odd r, alpha = {alpha:.3f} "
              f"(expect 2: 1/r^2 power law, LONG range)")

    # gapped: exponential (fit on the range where it's still above numerical floor)
    fit_r = np.array([1, 2, 3, 4, 5])
    cm_fit = C_m[fit_r]
    xi = None
    if (cm_fit > 1e-15).all():
        xi = -1.0 / np.polyfit(fit_r, np.log(cm_fit), 1)[0]
        print(f"  gapped:  C(r) ~ exp(-r/xi), xi = {xi:.2f} (exponential, SHORT range)")
    else:
        print("  gapped:  C(r) decays to <1e-15 within r<=5 (exponential, SHORT range)")

    summary = {
        "L": L,
        "gapless_C_odd": {str(r): float(C_g[r]) for r in odd},
        "gapped_C": {str(r): float(C_m[r]) for r in fit_r},
        "gapless_alpha": float(alpha),
        "gapped_xi": None if xi is None else float(xi),
        "conclusion": "Gapless H -> C(r) ~ 1/r^2 (power law, LONG range). Gapped H -> "
                      "exponential (SHORT range). Long-range 1/r requires a gapless "
                      "(massless) spectrum = continuous spectrum of log(Delta), which "
                      "finite D does not have => long-range lives in R, not D.",
    }
    out = ROOT / "experiments" / "exp_bridge_B_spectral_gap_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
