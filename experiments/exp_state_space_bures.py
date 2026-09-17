"""State-space Bures geometry: does "curvature on the state space" give a
computable, point-varying curved spacetime?  (Layer 1 of the "future mathematics"
program, per the new-session brief.)

PROBLEM RESTATED (the wall, precisely).  Long-range gravity needs LOCAL
diffeomorphism invariance, but the self-referential framework only has global
unitary D -> U D U^dag and outer modular automorphisms Out(R); "Out(R) = Diff"
is Connes's UNPROVEN analogy.  The wall = "generate differential structure from
measure" (the MASA of the II_1 factor gives L^inf(X, mu), not C^inf(M)).

THE TURN (this probe).  Do NOT ask for geometry ON R (no points, no tangent
vectors, no compact D).  Ask for geometry on the STATE SPACE S(R), which IS a
convex set and carries the Bures metric.  Effective metric:

    g_eff(x)_{mu nu} = d^2/dx^mu dx^nu  d_B(rho_x, rho_{x+dx})^2

This is computable.  The question this probe answers HONESTLY:

    Q1. Is the Bures metric on the FULL state space maximally symmetric
        (constant curvature), i.e. NO intrinsic point-to-point "curvature"?
    Q2. Does a 1-parameter family rho_x give flat (R=0) geometry?
    Q3. Does a 2-parameter family rho_{x,y} give NON-constant curvature --
        and if so, where does that curvature COME from?

HONEST BOUNDARY (stated up front).  This computes geometry on state space, not
on spacetime.  The two identification gaps ("state = position", "state-space
geometry = physical geometry") are NOT resolved by Layer 1; they are RELOCATED.
The results below show precisely WHERE the wall now sits.

Code: `py -m experiments.exp_state_space_bures`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAULI = np.array([
    [[0, 1], [1, 0]],
    [[0, -1j], [1j, 0]],
    [[1, 0], [0, -1]],
], dtype=complex)


# ----------------------------------------------------------------------------
# Bures distance
# ----------------------------------------------------------------------------
def bures2_qubit(r1, r2):
    """Closed-form squared Bures distance between two qubit states.

    rho_i = (1/2)(I + r_i . sigma),  |r_i| <= 1.
    F(r1,r2) = (1/2)[1 + r1.r2 + sqrt((1-|r1|^2)(1-|r2|^2))]
    d_B^2 = 2(1 - sqrt(F)).
    """
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    n1, n2 = r1 @ r1, r2 @ r2
    F = 0.5 * (1.0 + r1 @ r2 + np.sqrt(max(0.0, (1 - n1) * (1 - n2))))
    F = np.clip(F, 0.0, 1.0)
    return 2.0 * (1.0 - np.sqrt(F))


def bures2_general(rho1, rho2):
    """Squared Bures distance for arbitrary density matrices (matrix sqrt)."""
    s = sqrtm(rho1)
    F = np.real(np.trace(sqrtm(s @ rho2 @ s)))
    return 2.0 * (1.0 - np.clip(F, 0.0, 1.0))


def rho_qubit(r):
    r = np.asarray(r, float)
    return 0.5 * (np.eye(2) + r[0] * PAULI[0] + r[1] * PAULI[1] + r[2] * PAULI[2])


# ----------------------------------------------------------------------------
# closed-form qubit Bures metric (for cross-check)
# ----------------------------------------------------------------------------
def metric_qubit_closed(r):
    """g_mu nu = (1/4)[delta_mu nu + r_mu r_nu/(1-|r|^2)]  at Bloch vector r."""
    r = np.asarray(r, float)
    n = r @ r
    return 0.25 * (np.eye(3) + np.outer(r, r) / (1.0 - n))


def metric_fd(rho_of_x, x0, dims, delta=1e-3):
    """Finite-difference metric from the Bures distance (works for ANY family).

    rho_of_x: callable mapping a numpy array x (length dims) -> density matrix.
    g_mu nu = d^2/dx^mu dx^nu [d_B(rho(x0), rho(x0+dx))^2] / 2  (central).
    """
    g = np.zeros((dims, dims))
    for mu in range(dims):
        for nu in range(dims):
            if mu == nu:
                # central 2nd derivative along e_mu
                p = np.array(x0, float); p[mu] += delta
                m = np.array(x0, float); m[mu] -= delta
                d2 = (bures2_general(rho_of_x(x0), rho_of_x(p))
                      - 2.0 * bures2_general(rho_of_x(x0), rho_of_x(x0))
                      + bures2_general(rho_of_x(x0), rho_of_x(m))) / delta ** 2
                g[mu, nu] = 0.5 * d2
            else:
                pp = np.array(x0, float); pp[mu] += delta; pp[nu] += delta
                pm = np.array(x0, float); pm[mu] += delta; pm[nu] -= delta
                mp = np.array(x0, float); mp[mu] -= delta; mp[nu] += delta
                mm = np.array(x0, float); mm[mu] -= delta; mm[nu] -= delta
                mixed = (bures2_general(rho_of_x(x0), rho_of_x(pp))
                         - bures2_general(rho_of_x(x0), rho_of_x(pm))
                         - bures2_general(rho_of_x(x0), rho_of_x(mp))
                         + bures2_general(rho_of_x(x0), rho_of_x(mm))) / (4 * delta ** 2)
                g[mu, nu] = 0.5 * mixed
    return 0.5 * (g + g.T)


# ----------------------------------------------------------------------------
# Gaussian curvature of a DIAGONAL 2D metric  ds^2 = E dx^2 + G dy^2
# ----------------------------------------------------------------------------
def gauss_curvature_diag(E, G, x, y, dx, dy):
    """K = -1/(2 sqrt(EG)) [ d_x(G_x/sqrt(EG)) + d_y(E_y/sqrt(EG)) ].
    E, G: 2D arrays on the (x, y) grid."""
    dEx = np.gradient(E, dx, axis=0)
    dGy = np.gradient(G, dy, axis=1)
    s = np.sqrt(E * G)
    t1 = dGy / s          # G_x / sqrt(EG)   (gradient axis 0 = x)
    t2 = dEx / s          # E_y / sqrt(EG)   (gradient axis 1 = y)
    dt1 = np.gradient(t1, dx, axis=0)
    dt2 = np.gradient(t2, dy, axis=1)
    K = -1.0 / (2.0 * s) * (dt1 + dt2)
    return K


# ----------------------------------------------------------------------------
# Part A: Bures distance closed-form vs matrix-sqrt (sanity)
# ----------------------------------------------------------------------------
def partA():
    print("=== Part A. Bures distance: closed form vs matrix sqrt (qubit) ===")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(5):
        r1 = rng.uniform(-1, 1, 3); r1 *= 0.9 / np.linalg.norm(r1)
        r2 = rng.uniform(-1, 1, 3); r2 *= 0.9 / np.linalg.norm(r2)
        d_closed = bures2_qubit(r1, r2)
        d_matrix = bures2_general(rho_qubit(r1), rho_qubit(r2))
        worst = max(worst, abs(d_closed - d_matrix))
    print(f"    max |closed - matrix| = {worst:.2e}   (should be ~1e-15)")
    return {"max_dev": worst}


# ----------------------------------------------------------------------------
# Part B: metric on the qubit Bloch ball -- closed form vs finite difference
# ----------------------------------------------------------------------------
def partB():
    print()
    print("=== Part B. Bures metric on Bloch ball: closed form vs finite diff ===")
    r = np.array([0.30, -0.20, 0.45])
    g_closed = metric_qubit_closed(r)
    rho_fn = lambda x: rho_qubit(x)
    g_fd = metric_fd(rho_fn, r, 3)
    dev = np.max(np.abs(g_closed - g_fd))
    print(f"    r = {r}")
    print(f"    g (closed)  = {np.round(g_closed, 4).tolist()}")
    print(f"    g (fin.diff)= {np.round(g_fd, 4).tolist()}")
    print(f"    max dev     = {dev:.2e}")
    return {"r": r.tolist(), "max_dev": dev}


# ----------------------------------------------------------------------------
# Part C: curvature of the FULL state space (equatorial slice) -- constant?
# ----------------------------------------------------------------------------
def partC():
    print()
    print("=== Part C. curvature of the FULL state space: hemisphere of S^3 ===")
    print("    qubit Bloch ball metric:  ds^2 = (1/4)[dr^2/(1-r^2) + r^2 dOmega^2]")
    print("    substitute r = sin(chi), chi in [0, pi/2]:")
    print("      ds^2 = (1/4)[dchi^2 + sin^2(chi) dOmega^2]  = round S^3 of radius 1/2,")
    print("      restricted to ONE hemisphere (chi in [0, pi/2]).")
    print("    => sectional curvature K = 1/R^2 = 4, CONSTANT (maximally symmetric).")
    print("    => the FULL state space has NO intrinsic point-to-point curvature.")
    # clean numeric confirmation on interior points only (finite-diff of K is
    # ill-conditioned near r->0 (G->0) and r->1 (E->inf); interior is ~4)
    r_grid = np.linspace(0.30, 0.60, 400)
    E = 1.0 / (4.0 * (1.0 - r_grid ** 2))
    G = (r_grid ** 2) / 4.0
    s = np.sqrt(E * G)
    dG = np.gradient(G, r_grid[1] - r_grid[0])
    Knum = -1.0 / (2.0 * s) * np.gradient(dG / s, r_grid[1] - r_grid[0])
    print(f"    numeric K on interior r in [0.30,0.60]: mean={Knum.mean():.3f} "
          f"(analytic = 4; finite-diff noise from E,G diverging at the boundary)")
    return {"K_analytic": 4.0, "K_numeric_mean_interior": float(Knum.mean()),
            "hemisphere_of_S3": True, "constant_curvature": True}


# ----------------------------------------------------------------------------
# Part D: 1-parameter family (flat) vs 2-parameter family (non-constant K)
# ----------------------------------------------------------------------------
def partD():
    print()
    print("=== Part D. induced metric on a FAMILY rho_x (the actual proposal) ===")

    # D1. 1-parameter family -> 1D metric -> identically flat (R=0 always)
    print("  D1. 1-parameter family rho_x (a curve in state space):")
    print("      1D metric has NO curvature (every 1D metric is flat, R=0).")
    print("      => 'long-range' (translation along one x) is always FLAT.")

    # D2. 2-parameter family: a torus surface in the Bloch ball (non-degenerate,
    #     avoids the coordinate cusp of a longitude/latitude parametrization)
    print("  D2. 2-parameter family rho_{x,y} = (1/2)(I + R(x,y).sigma),")
    print("      R(x,y) = torus: ((R+r cos x)cos y, (R+r cos x)sin y, r sin x)")
    R0, r0 = 0.45, 0.18

    def vec(x, y):
        return np.array([(R0 + r0 * np.cos(x)) * np.cos(y),
                         (R0 + r0 * np.cos(x)) * np.sin(y),
                         r0 * np.sin(x)])

    # induced metric via pullback of the qubit Bures metric; the metric depends
    # only on x (meridian), so Gaussian curvature K is a function of x alone.
    x_grid = np.linspace(0, 2 * np.pi, 500)
    y0 = 0.3
    eps = 1e-6
    Exx_arr = np.zeros(len(x_grid)); Eyy_arr = np.zeros(len(x_grid))
    for i, x in enumerate(x_grid):
        r = vec(x, y0)
        g = metric_qubit_closed(r)
        dxr = (vec(x + eps, y0) - vec(x - eps, y0)) / (2 * eps)
        dyr = (vec(x, y0 + eps) - vec(x, y0 - eps)) / (2 * eps)
        Exx_arr[i] = dxr @ g @ dxr
        Eyy_arr[i] = dyr @ g @ dyr
    s = np.sqrt(Exx_arr * Eyy_arr)
    dx = x_grid[1] - x_grid[0]
    dEyy = np.gradient(Eyy_arr, dx)
    K_of_x = -1.0 / (2.0 * s) * np.gradient(dEyy / s, dx)

    print(f"      K(x) over x in [0, 2pi]:")
    print(f"        mean = {np.nanmean(K_of_x):+.4f}   std = {np.nanstd(K_of_x):.4f}")
    print(f"        min = {np.nanmin(K_of_x):+.4f}   max = {np.nanmax(K_of_x):+.4f}")
    varying = bool(np.nanstd(K_of_x) > 1e-2)
    print(f"      => K(x) {'VARIES with x (non-constant curvature)' if varying else 'is constant'}")

    print()
    print("    HONEST: this non-constant K(x) is the curvature of the EMBEDDING,")
    print("      i.e. it is ENTIRELY determined by the chosen family R(x), h(x).")
    print("      It is INPUT, not emergent from R.  The full state-space Bures")
    print("      metric (Part C) is maximally symmetric (constant K=4), so there")
    print("      is NO intrinsic preferred family.  Choosing the family = the")
    print("      identification gap 'state = position' (still open).")
    return {"K_mean": float(np.nanmean(K_of_x)), "K_std": float(np.nanstd(K_of_x)),
            "K_min": float(np.nanmin(K_of_x)), "K_max": float(np.nanmax(K_of_x)),
            "varying": varying}


# ----------------------------------------------------------------------------
# Part E: N=4 commuting thermal family -> Fisher metric (general machinery)
# ----------------------------------------------------------------------------
def partE():
    print()
    print("=== Part E. N=4 commuting thermal family -> Fisher metric ===")
    print("    rho_x = e^{-beta H(x)}/Z,  H(x) diagonal (commuting) -> Bures =")
    print("    Hellinger -> g(x) = (1/4) Fisher information = (1/4) sum (p_i')^2/p_i")

    def probs(x):
        # 4-level energies E_i(x), beta=1
        E = np.array([0.0, 1.0 + 0.5 * x, 2.0 + x * x, 3.0 + 0.3 * np.sin(x)])
        w = np.exp(-E)
        return w / w.sum()

    def rho_of_x(x):
        p = probs(x[0])
        return np.diag(p)

    x0 = 0.7
    g_fd = metric_fd(rho_of_x, np.array([x0]), 1, delta=1e-3)
    # closed form g(x) = (1/4) sum p_i'^2 / p_i
    eps = 1e-5
    p = probs(x0)
    p_plus = probs(x0 + eps); p_minus = probs(x0 - eps)
    dp = (p_plus - p_minus) / (2 * eps)
    g_closed = 0.25 * np.sum(dp ** 2 / p)
    print(f"    x0={x0}: g_fd={g_fd[0,0]:.6f}  g_closed={g_closed:.6f}")
    print(f"    => Bures metric on a commuting family = (1/4) x classical Fisher")
    print(f"       information (statistical manifold).  1-parameter -> flat.")
    return {"g_fd": float(g_fd[0, 0]), "g_closed": float(g_closed)}


def main():
    print("=== state-space Bures geometry: Layer 1 of the 'future mathematics' ===")
    print("    (the turn: geometry on S(R), not on R)")
    print()
    a = partA()
    b = partB()
    c = partC()
    d = partD()
    e = partE()

    print()
    print("=== verdict ===")
    print("  1. The Bures metric on the FULL state space is MAXIMALLY SYMMETRIC")
    print("     (constant curvature, a hemisphere of S^3 radius 1/2): no intrinsic")
    print("     point-to-point 'curvature'.  So state-space geometry does NOT by")
    print("     itself produce a position-varying curved spacetime.")
    print("  2. A 1-parameter family rho_x gives FLAT (R=0): one 'position' coordinate")
    print("     can never curve.")
    print("  3. A 2-parameter family rho_{x,y} DOES give non-constant curvature, but")
    print("     that curvature is the embedding's -- entirely INPUT via the chosen")
    print("     family.  The wall is RELOCATED, not removed: it now sits at")
    print("     'state = position' (which family rho_x?) and 'state-space geometry")
    print("     = physical geometry' (the two identification gaps).")
    print("  4. Layer 1 is a REAL, runnable anchor (computable curvature), and the")
    print("     cleanest way to say WHERE the wall now sits.  It does not touch the")
    print("     two walls in the 'new mathematics' (commutator vanishing / groupoid")
    print("     etale-ification); it is a GEOMETRIC reformulation of the same wall.")

    summary = {
        "turn": "geometry on state space S(R) (Bures metric), not on R",
        "partA_sanity": a,
        "partB_metric_verify": b,
        "partC_full_state_curvature": c,
        "partC_conclusion": "full state-space Bures metric is maximally symmetric "
                            "(constant K=4, hemisphere of S^3), no intrinsic point-varying curvature",
        "partD_family": d,
        "partD_conclusion": "1D family flat; 2D family non-constant K but INPUT via family "
                            "(the 'state = position' identification gap, still open)",
        "partE_N4": e,
        "verdict": "Layer 1 is a real computable anchor; it RELOCATES the wall to the two "
                   "identification gaps (state=position, state-space=physical geometry); it "
                   "does not dissolve the Connes wall ('from measure to differential structure').",
    }
    out = ROOT / "experiments" / "exp_state_space_bures_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
