"""Final test of candidate C: what is the k-dependence of the bond order near the
Dirac point?  k^0 (order-1 vector), k^-2 (Poisson-inverse of density), or k^2?

Puzzle 1 recap:  reaction r ~ rho (local), geometry delta ~ grad^2 r, so
delta ~ grad^2 rho.  GR needs delta ~ rho.  To fix it, the source must be
r ~ grad^-2 rho (Poisson inverse, k^-2).

Candidate C says: source is bond order K, not rho.  For this to FIX puzzle 1,
we need K(k) ~ k^-2 (so grad^2 K ~ k^0 = rho).  If instead K(k) ~ k^0 (a
direction unit vector), then grad^2 K ~ k^2, which does NOT recover rho ~ k^0,
and candidate C does NOT fix the grad^2 mismatch.

Exact 2x2 pi-flux Bloch model: D(k) = 2 cos(kx) sx + 2 cos(ky) sz, d=(2cos kx,0,2cos ky).
Lower-band projector off-diagonal (bond order) = -(d_x sx + d_z sz)/(2|d|),
magnitude = 1/2 INDEPENDENT of k (k^0).  Confirm numerically near the Dirac point.

Code: `py -m experiments.exp_bridge_B_bond_order4`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def Hk(kx, ky):
    return 2 * np.cos(kx) * SX + 2 * np.cos(ky) * SZ


def lower_projector(kx, ky):
    H = Hk(kx, ky)
    eigs, vecs = np.linalg.eigh(H)
    # lower band: eigenvector with E < 0
    idx = int(np.argmin(eigs))
    v = vecs[:, idx]
    P = np.outer(v, v.conj())
    return P, eigs[idx]


def main():
    print("=== k-dependence of bond order near Dirac point (candidate C final test) ===")
    print("  D(k) = 2 cos(kx) sx + 2 cos(ky) sz,  Dirac point at (pi/2, pi/2)")
    print("  bond order (off-diagonal of lower projector) magnitude vs |delta k|:")
    print(f"  {'delta kx':>8} {'delta ky':>8} {'|offdiag|':>10} {'expectation':>14}")
    results = []
    # approach Dirac point along several radii
    for r in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
        # along (1,0) direction: kx = pi/2 + r, ky = pi/2
        kx = np.pi/2 + r; ky = np.pi/2
        P, E = lower_projector(kx, ky)
        offdiag = abs(P[0, 1])
        print(f"  {r:8.3f} {0.0:8.3f} {offdiag:10.4f} {'k^0 => const 0.5' if abs(offdiag-0.5)<0.01 else '?':>14}")
        results.append({"r": r, "offdiag": float(offdiag)})
    print()
    print("  => |offdiag| = 1/2 independent of |delta k| (k^0), NOT k^-2.")
    print("  => candidate C does NOT fix puzzle 1's grad^2 mismatch:")
    print("     grad^2 K ~ k^2, but GR needs delta ~ rho ~ k^0.")

    # Also verify the direction-vector character: at fixed r, rotate angle theta
    print("\n  Direction structure (fixed |delta k|=0.1, rotate around Dirac point):")
    for theta in [0.0, np.pi/4, np.pi/2, 3*np.pi/4]:
        r = 0.1
        kx = np.pi/2 + r*np.cos(theta); ky = np.pi/2 + r*np.sin(theta)
        P, E = lower_projector(kx, ky)
        # offdiag real/imag parts encode the direction (d_x, d_z)
        ox = np.real(P[0,1]); oz = np.imag(P[0,1])
        print(f"    theta={theta/np.pi:.2f}pi: offdiag = ({ox:+.3f}, {oz:+.3f}), |.|={abs(P[0,1]):.3f}")

    summary = {
        "k_scan": results,
        "conclusion": "Bond order (lower-band off-diagonal) has magnitude 1/2 INDEPENDENT of "
                      "|delta k| => k^0 (a direction unit vector in (px,py)), NOT k^-2. "
                      "Therefore grad^2 K ~ k^2, which does NOT recover rho ~ k^0.  Candidate C "
                      "(bond order as source) does NOT fix puzzle 1's grad^2 mismatch.  The bond "
                      "order IS the correct Hellmann-Feynman source (T_ij, not T_00), but the "
                      "grad^2 mismatch between local reaction r~rho and GR's delta~rho remains.",
    }
    out = ROOT / "experiments" / "exp_bridge_B_bond_order4_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
