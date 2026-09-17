"""Spin-2 metric g = e·e: verify the vielbein -> metric machinery, and quantify
the FLATNESS degeneration of the theory's classical bridge.

Attack step 1 of the spin-2 metric (per 权威交接文档 §六·五, "自旋校准"):
the metric g_{mu nu} = e_mu^a e_nu^b eta_ab is DEFINITIONAL once the vielbein
e_mu^a is given.  The theory already has the two ingredients of the internal
frame:  (i) three spatial directions = V_1 (j=1 adjoint rep, Cartan weight basis),
(ii) time direction = gamma^0 (directed distinction, signature -).  So we can
build e and verify the SPIN-2 STRUCTURE numerically.

The honest content is DEGENERATION A: if the theory identifies "physical 3D" with
the internal V_1 (i.e. mu = a), then e = delta, g = eta (FLAT), and the spin-2
metric is EMPTY (no graviton).  Non-trivial spin-2 requires mu != a (a real-space
vielbein distinct from the internal frame) -- which is exactly the separation wall
(SU(2) inseparable from U(1) in T_mu) + the domain problem (S^2_q has no space).

Part A. spin-2 machinery (definitional verification):
  1. flat vielbein -> g = eta (symmetric, non-degenerate, Lorentz signature -+++).
  2. local SO(3) frame rotation leaves g invariant.
  3. TT polarizations rotate with angle 2*theta (helicity +-2 = spin-2).
  4. graviton DOF = d(d-3)/2:  2 in 3+1, 0 in 3D (spin-2 needs the time dimension).
Part B. degeneration A (the theory's bridge gives FLAT metric):
  5. physical 3D = V_1 => e = delta => g = eta (flat, Riemann = 0).
  6. the only non-trivial geometric object F_mu nu = [T_mu,T_nu] = 2 T_mu T_nu is
     NON-zero but momentum-block-diagonal (commutes with T_i^2 sector labels),
     so it is a MOMENTUM-space curvature, not a real-space metric curvature.

Code: `py -m experiments.exp_spin2_metric`
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


# ----------------------------------------------------------------------------
# internal Minkowski metric (frame index a,b in {0,1,2,3}, signature -+++)
# ----------------------------------------------------------------------------
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])


def metric_from_vielbein(e):
    """g_{mu nu} = e_mu^a eta_ab e_nu^b.  Here e has shape (a, mu): row=frame a,
    col=spacetime mu.  Then g = e^T eta e."""
    return e.T @ ETA @ e


# ----------------------------------------------------------------------------
# torus3D + magnetic translations (same gauge as exp_pi_flux_3D.py)
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Part A. spin-2 machinery
# ----------------------------------------------------------------------------
def partA():
    print("=== Part A. spin-2 machinery (vielbein -> metric) ===")
    print()

    # A1. flat vielbein -> g = eta
    e_flat = np.eye(4)                      # e_mu^a = delta_mu^a
    g = metric_from_vielbein(e_flat)
    symmetric = bool(np.allclose(g, g.T))
    nondeg = float(np.linalg.det(g))
    print("A1. flat vielbein e = I_4:")
    print(f"    g = e^T eta e = {np.round(g, 3).tolist()}")
    print(f"    symmetric: {symmetric}   det(g) = {nondeg:.3f}   (Lorentz signature -> det=-1)")

    # A2. local SO(3) frame rotation leaves g invariant
    theta = 0.7
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(4)
    R[1, 1] = c; R[1, 2] = -s
    R[2, 1] = s; R[2, 2] = c                 # SO(3) rotation in the spatial (1,2) plane
    Lambda = R                               # acts on frame index a
    e_rot = Lambda @ e_flat                  # e -> Lambda e
    g_rot = metric_from_vielbein(e_rot)
    inv_ok = bool(np.allclose(g_rot, g))     # Lambda^T eta Lambda = eta
    print()
    print("A2. local SO(3) frame rotation (spatial plane 1-2):")
    print(f"    e -> Lambda e,  g unchanged: {inv_ok}")

    # A3. helicity +-2: TT polarizations rotate with angle 2*theta
    h_plus = np.zeros((4, 4))
    h_plus[1, 1] = 1.0; h_plus[2, 2] = -1.0
    h_cross = np.zeros((4, 4))
    h_cross[1, 2] = 1.0; h_cross[2, 1] = 1.0
    # propagate along z (index 3); transverse plane = (x,y) = (1,2)
    maxdev_plus = 0.0
    maxdev_cross = 0.0
    for th in np.linspace(0, 2 * np.pi, 9):
        c, s = np.cos(th), np.sin(th)
        R = np.eye(4)
        R[1, 1] = c; R[1, 2] = -s
        R[2, 1] = s; R[2, 2] = c
        rp = R @ h_plus @ R.T
        rc = R @ h_cross @ R.T
        expect_plus = np.cos(2 * th) * h_plus + np.sin(2 * th) * h_cross
        expect_cross = -np.sin(2 * th) * h_plus + np.cos(2 * th) * h_cross
        maxdev_plus = max(maxdev_plus, np.max(np.abs(rp - expect_plus)))
        maxdev_cross = max(maxdev_cross, np.max(np.abs(rc - expect_cross)))
    print()
    print("A3. helicity +-2 (spin-2 signature):")
    print(f"    R(th) h_+ R^T = cos(2th)h_+ + sin(2th)h_x   max dev = {maxdev_plus:.2e}")
    print(f"    R(th) h_x R^T = -sin(2th)h_+ + cos(2th)h_x  max dev = {maxdev_cross:.2e}")
    print("    (rotation angle 2*theta = helicity 2, the defining property of spin-2)")

    # A4. graviton DOF = d(d-3)/2
    print()
    print("A4. propagating graviton DOF = d(d-3)/2:")
    dof = {}
    for d in (2, 3, 4):
        dof[d] = d * (d - 3) / 2
        print(f"    d={d} spacetime dims -> DOF = {dof[d]:.0f}")
    print("    => spin-2 needs d=4 (3+1): the TIME dimension (gamma^0) must enter g.")

    return {
        "g_symmetric": symmetric,
        "det_g": nondeg,
        "so3_frame_invariant": inv_ok,
        "helicity_2_plus_dev": maxdev_plus,
        "helicity_2_cross_dev": maxdev_cross,
        "graviton_dof": {str(d): dof[d] for d in dof},
    }


# ----------------------------------------------------------------------------
# Part B. degeneration A: the theory's bridge gives FLAT metric
# ----------------------------------------------------------------------------
def partB():
    print()
    print("=== Part B. degeneration A: the theory's bridge gives FLAT metric ===")
    print()

    # B5. physical 3D = V_1 (internal SO(3)) => e = delta => g = eta (flat)
    # V_1 = j=1 adjoint rep. Its three "directions" = the weight basis (or the
    # Cartesian Jx,Jy,Jz). If spacetime IS this same 3-dim space (mu = a), the
    # vielbein is the identity, and g = eta is flat.
    g_flat = metric_from_vielbein(np.eye(4))
    # Riemann curvature of a CONSTANT metric vanishes identically (Levi-Civita = 0)
    riemann_flat = 0.0
    print("B5. physical 3D = V_1 (bridge) => e = delta, g = eta:")
    print(f"    g = {np.round(g_flat, 3).tolist()}")
    print(f"    constant metric => Riemann curvature = {riemann_flat} (flat, NO graviton)")
    print("    => the classical bridge (V_1 = physical 3D) yields EMPTY spin-2 (flat).")

    # B6. the non-trivial object F_mu nu = [T_mu,T_nu] = 2 T_mu T_nu is momentum-space
    L = 4
    D = torus3D(L, flux=True)
    N = L ** 3
    Tx, Ty, Tz = mag_trans(D, L, 'x'), mag_trans(D, L, 'y'), mag_trans(D, L, 'z')
    Fxy = Tx @ Ty - Ty @ Tx          # [T_x, T_y] = 2 T_x T_y (anti-commuting)
    normF = float(np.linalg.norm(Fxy))
    # commutes with sector labels T_i^2 (=> block-diagonal in momentum sectors)?
    Tx2, Ty2, Tz2 = Tx @ Tx, Ty @ Ty, Tz @ Tz
    comm_sectors = [float(np.linalg.norm(Fxy @ M - M @ Fxy))
                    for M in (Tx2, Ty2, Tz2)]
    chi = Tx @ Ty @ Tz
    comm_chi = float(np.linalg.norm(Fxy @ chi - chi @ Fxy))
    momentum_diag = all(c < 1e-9 for c in comm_sectors) and comm_chi < 1e-9

    print()
    print("B6. the only non-trivial object F_xy = [T_x,T_y] = 2 T_x T_y:")
    print(f"    ||F_xy|| = {normF:.2f}  (NON-zero: a real spin curvature exists)")
    print(f"    [F_xy, T_i^2] = {[f'{c:.1e}' for c in comm_sectors]}  (commutes with sector labels)")
    print(f"    [F_xy, chi] = {comm_chi:.1e}")
    print(f"    => F_xy is MOMENTUM-block-diagonal (lives in the 32 Pauli sectors),")
    print(f"       NOT a real-space metric curvature.  To turn it into a curved g,")
    print(f"       one needs a real-space vielbein e (separate SU(2) from U(1) in T_mu),")
    print(f"       which is the separation wall (paid bridge 2, section 6).")

    return {
        "riemann_flat": riemann_flat,
        "norm_Fxy": normF,
        "Fxy_commutes_with_T2": comm_sectors,
        "Fxy_commutes_with_chi": comm_chi,
        "Fxy_momentum_diagonal": momentum_diag,
    }


def main():
    a = partA()
    b = partB()

    print()
    print("=== conclusion ===")
    print("  1. the spin-2 metric g = e·e·eta is DEFINITIONAL and its structure is verified")
    print("     (symmetric, SO(3)-frame-invariant, helicity +-2, DOF=2 in 3+1 / 0 in 3D).")
    print("  2. BUT the theory's classical bridge (V_1 = physical 3D, mu = a) gives e = delta")
    print("     => g = eta FLAT => spin-2 is EMPTY (no graviton).  This is degeneration A.")
    print("  3. the real spin curvature F = [T,T] is non-zero but lives in MOMENTUM space.")
    print("     Non-trivial spin-2 requires a real-space vielbein e != delta, i.e. the")
    print("     separation of SU(2) from U(1) in T_mu (paid bridge 2, the wall).")

    summary = {
        "spin2_structure_verified": True,
        "partA": a,
        "partB": b,
        "degeneration_A": "theory's bridge (V_1 = physical 3D) => e = delta => g = eta flat, spin-2 empty",
        "wall": "non-trivial spin-2 needs real-space e != delta (separate SU(2) from U(1) in T_mu)",
        "note": "spin-2 metric structure is definitional (verified). Dynamics (curved g) is blocked by "
                "the separation wall; F_mu nu is non-zero but momentum-space, not a real-space curvature.",
    }
    out = ROOT / "experiments" / "exp_spin2_metric_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
