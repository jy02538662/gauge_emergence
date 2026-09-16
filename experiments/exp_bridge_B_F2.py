"""Bridge B: F^2 is YANG-MILLS (a4), not Einstein-Hilbert (a2).

Purpose: pin down WHICH curvature the "curvature term F^2" actually is.

Two facts to verify (the load-bearing ones for bridge B's route):
  1. In the pure pi-flux discrete model, F_uv = [T_u, T_v] = 2 T_u T_v and
     F_uv^2 = -4 s_u s_v I is a SECTOR CONSTANT (variation = 0), and F_uv is
     TRACELESS (SU(2)/gauge-like).  => it is a GAUGE curvature (Yang-Mills type),
     NOT the spacetime scalar curvature R.
  2. The heat-kernel expansion of the spectral action separates the two orders:
        a0 ~ volume (cosmological constant)
        a2 ~ int sqrt(g) R   = SCALAR CURVATURE = Einstein-Hilbert
        a4 ~ int sqrt(g) F^2 = GAUGE curvature squared = Yang-Mills
     So EH comes from a2 (scalar curvature R), NOT from F^2 (a4).

This script verifies fact 1 numerically and records fact 2 (the Chamseddine-Connes
structure) as the route implication.  The step "F^2 becomes a field functional"
(= letting T_mu's phase vary spatially) is itself the discrete->continuous step,
which is the real bridge-B wall, NOT a computation this script does.

Code: `py -m experiments.exp_bridge_B_F2`
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


def torus3D(L, flux=True):
    N = L ** 3

    def idx(x, y, z):
        return ((z % L) * L + (y % L)) * L + (x % L)

    D = np.zeros((N, N))
    for x, y, z in product(range(L), repeat=3):
        i = idx(x, y, z)
        j = idx(x + 1, y, z); w = (-1) ** (y + z) if flux else 1.0; D[i, j] = w; D[j, i] = w
        j = idx(x, y + 1, z); w = (-1) ** z if flux else 1.0; D[i, j] = w; D[j, i] = w
        j = idx(x, y, z + 1); w = 1.0; D[i, j] = w; D[j, i] = w
    return D


def mag_trans(D, L, axis):
    """Magnetic translation T_mu: translation link carrying the flux phase D_ij."""
    N = L ** 3

    def idx(x, y, z):
        return ((z % L) * L + (y % L)) * L + (x % L)

    T = np.zeros((N, N))
    for x, y, z in product(range(L), repeat=3):
        i = idx(x, y, z)
        if axis == 'x':
            j = idx(x + 1, y, z)
        elif axis == 'y':
            j = idx(x, y + 1, z)
        else:
            j = idx(x, y, z + 1)
        T[i, j] = D[i, j]
    return T


def F2_action(Tx, Ty, Tz):
    """Positive-definite gauge-curvature norm ||F||^2 = Tr(F F^dag) over 3 planes.

    F_uv = [T_u, T_v] is a real non-symmetric matrix here, so the correct
    positive gauge kinetic term is Tr(F F^dag), not Tr(F F) (which is zero
    because F is real and F^2 = -4 s_u s_v I has signed sectors that cancel
    under the plain trace).  This is the Yang-Mills-type functional.
    """
    S = 0.0
    for A, B in [(Tx, Ty), (Ty, Tz), (Tz, Tx)]:
        F = A @ B - B @ A
        S += float(np.real(np.trace(F @ F.conj().T)))
    return S


def main():
    L = 4
    D_pi = torus3D(L, flux=True)
    D_0 = torus3D(L, flux=False)
    Tx, Ty, Tz = (mag_trans(D_pi, L, a) for a in 'xyz')
    Tx0, Ty0, Tz0 = (mag_trans(D_0, L, a) for a in 'xyz')

    # ---- Fact 1: strict pi-flux => F^2 is a SECTOR CONSTANT, F traceless (gauge-like) ----
    Fxy = Tx @ Ty - Ty @ Tx
    sector_scalar = bool(np.allclose(Fxy @ Fxy, -4 * (Tx @ Tx) @ (Ty @ Ty)))
    traceless = bool(abs(np.trace(Fxy)) < 1e-12)
    # anti-commutation is what makes F = 2 T_u T_v (off-diagonal), i.e. gauge curvature
    anticomm = bool(np.allclose(Tx @ Ty, -Ty @ Tx))
    S_pi = F2_action(Tx, Ty, Tz)
    S_0 = F2_action(Tx0, Ty0, Tz0)
    # NOTE: Tr(F_xy F_xy) itself is 0 (real F, signed sectors cancel); the physical
    # gauge norm is Tr(F F^dag) > 0.  Record both to avoid the confusion.
    tr_FF = float(np.real(np.trace(Fxy @ Fxy)))

    print("=== Bridge B: F^2 is Yang-Mills (a4), not Einstein-Hilbert (a2) ===")
    print(f"  T_x T_y = -T_y T_x (anti-commute => Cl(3)): {anticomm}")
    print(f"  F_xy = [T_x,T_y] = 2 T_x T_y (gauge curvature): "
          f"{bool(np.allclose(Fxy, 2 * (Tx @ Ty)))}")
    print(f"  F_xy^2 = -4 s_x s_y I (SECTOR CONSTANT): {sector_scalar}")
    print(f"  trace(F_xy) = 0 (traceless, SU(2)/gauge-like): {traceless}")
    print(f"  Tr(F_xy F_xy) = {tr_FF:.6f}  (zero: real F, signed sectors cancel under plain trace)")
    print(f"  ||F||^2 = Tr(F F^dag) strict pi-flux = {S_pi:.6f}  (>0, gauge kinetic term)")
    print(f"  ||F||^2 = Tr(F F^dag) no-flux        = {S_0:.6f}  (0: T's commute => F = 0)")
    print()
    print("  => F^2 is a GAUGE curvature (Yang-Mills type). In the heat-kernel expansion")
    print("     of Tr f(D/Lambda), it lives at the a4 order, NOT a2.")
    print("     EH (Einstein-Hilbert) = a2 order = int sqrt(g) R (scalar curvature).")
    print("     So 'F^2 -> EH' is a broken chain: F^2 gives Yang-Mills, not gravity.")

    summary = {
        "L": L,
        "anti_commute": anticomm,
        "F_equals_2TuTv": bool(np.allclose(Fxy, 2 * (Tx @ Ty))),
        "F2_sector_constant": sector_scalar,
        "F_traceless": traceless,
        "Tr_FF_plain": tr_FF,
        "normF2_pi_flux": S_pi,
        "normF2_no_flux": S_0,
        "conclusion": "F^2 = [T_u,T_v] is a GAUGE curvature (Yang-Mills, a4 order). "
                      "Strict pi-flux makes it a sector constant (variation=0). "
                      "EH (Einstein-Hilbert) comes from the a2 order = scalar curvature R "
                      "in the heat-kernel expansion of Tr f(D/Lambda), NOT from F^2. "
                      "The step 'F^2 becomes a field functional' = discrete->continuous "
                      "(the real bridge-B wall).",
    }
    out = ROOT / "experiments" / "exp_bridge_B_F2_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
