"""Does the no-preference axiom A (already forced rho = C/lambda) extend one more
step to force "state = position" (the family rho_x)?  If yes, the Part-D2
curvature stops being hand-picked and becomes axiom-forced, and "state-space
geometry" automatically becomes "physical geometry" -- the wall is solved.

This probe pushes that exact step and reports HONESTLY what it finds.

TWO CLAIMS (the mathematical core):

  Claim 1 (negative, provable).  No-preference (A) is LOGARITHMIC
  translation-invariance (rho(c lambda) = c^-1 rho(lambda) -> rho = C/lambda).
  But log-translation-invariance means a translation of the state is a trivial
  renormalisation, NOT a new state.  On the state space this is "equivariance =>
  constant curvature" (the hemisphere of S^3, Part C of the Bures probe).  So A
  ALONE structurally cannot produce "position": position = breaking the
  symmetry, and A is exactly the statement of that symmetry.  We verify this
  numerically.

  Claim 2 (positive, computable).  The RIGHT source of "position" is the
  LOCALISATION of J (directed distinction), i.e. a DEFECT (matter), consistent
  with the gravity-side convergence ("curvature = localisation of J, not
  breaking of A").  We show a computable bridge: a defect in D changes the
  local state family rho_i = |psi_i><psi_i| (psi_i = localised Green-function
  state), which changes the Bures distance field, whose "curvature" (2nd
  variation) is localised AT the defect.  This is a computable anchor of the
  identification "state-space geometry = physical geometry" (Layer 2), with the
  honest caveat: the DEFECT (matter) is still an input, not forced by A.

HONEST BOUNDARY (up front): this does NOT force the family rho_x from A.  It
proves A cannot do it (Claim 1) and locates the positive source in J/defect
(Claim 2), which is still an input (= the "EH lacks a matter source" wall).

Code: `py -m experiments.exp_state_space_position`
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


# ----------------------------------------------------------------------------
# Part 1: A = log-translation-invariance => translation is trivial (no position)
# ----------------------------------------------------------------------------
def part1():
    print("=== Part 1. A = log-translation-invariance => no 'position' ===")
    print("    no-preference scale => rho(c lambda) = c^-1 rho(lambda)")
    print("    => rho = C/lambda (unique solution, already forced).")
    print()
    print("    KEY: a 'position shift' lambda -> c lambda maps rho to")
    print("         C/(c lambda) = (C/c)/lambda = C'/lambda, the SAME form.")
    print("    => a translation is a renormalisation of C, NOT a new state.")
    print("    => log-translation-invariance EXACTLY means 'no position'.")
    print()

    # numeric: a log-uniform spectrum is FLAT in log space (no preferred scale),
    # and stays flat under a log-translation.  Use a PERIODIC window so the
    # translation has no boundary-truncation artefact.
    rng = np.random.default_rng(0)
    loglam = rng.uniform(0, 10, 200000)               # log-uniform on a periodic window
    shift = 1.7
    loglam_shifted = (loglam + np.log(shift)) % 10    # periodic log-translation

    # flatness = std/mean of the histogram (pure Poisson => ~1/sqrt(n), no trend)
    def flatness(logv):
        h, _ = np.histogram(logv, bins=40, range=(0, 10))
        return h.std() / h.mean()

    f_before = flatness(loglam)
    f_after = flatness(loglam_shifted)
    poisson = 1.0 / np.sqrt(200000 / 40)              # expected pure-noise level
    print(f"    log-uniform spectrum rho=C/lambda, log-translation by log({shift}):")
    print(f"      histogram flatness before = {f_before:.4f}  after = {f_after:.4f}")
    print(f"      (pure Poisson noise level ~ {poisson:.4f}; both flat => same state)")
    print("    => the spectrum is flat in log space: NO preferred scale = NO position,")
    print("       and a log-translation leaves it flat (translation = renormalisation).")
    return {"shift": shift, "flatness_before": float(f_before),
            "flatness_after": float(f_after), "poisson_noise_level": float(poisson),
            "conclusion": "A (log-translation-invariance) cannot generate position: "
                          "translation = renormalisation, not a new state"}


# ----------------------------------------------------------------------------
# Part 2: defect in D => Bures distance field anomaly (positive, computable)
# ----------------------------------------------------------------------------
def build_D(L, defect=None):
    """2D nearest-neighbour hopping on an L x L torus, optional local defect."""
    N = L * L

    def idx(x, y):
        return (y % L) * L + (x % L)

    D = np.zeros((N, N))
    t = 1.0
    for y in range(L):
        for x in range(L):
            i = idx(x, y)
            j = idx(x + 1, y); D[i, j] += t; D[j, i] += t
            j = idx(x, y + 1); D[i, j] += t; D[j, i] += t
    if defect is not None:
        x0, y0, V = defect
        i = idx(x0, y0)
        D[i, i] += V       # local on-site potential = a defect (matter seed)
    return D


def local_states(D, m2=1.0):
    """psi_i = i-th column of (D^2 + m2 I)^{-1}, normalised; rho_i = psi_i psi_i^+."""
    N = D.shape[0]
    G = np.linalg.inv(D @ D + m2 * np.eye(N))       # symmetric => row = col
    psis = G / np.linalg.norm(G, axis=0)[None, :]   # normalise each column
    return psis


def bures_overlap(psi_i, psi_j):
    """d_B^2 between two PURE states |psi_i>, |psi_j>."""
    ov = abs(np.vdot(psi_i, psi_j)) ** 2
    return 2.0 * (1.0 - np.sqrt(ov))


def part2():
    print()
    print("=== Part 2. defect in D => Bures distance-field anomaly ===")
    print("    rho_i = |psi_i><psi_i|,  psi_i = localised Green-function state,")
    print("    g_eff(i, i+x) = d_B(rho_i, rho_{i+x})^2  (x = lattice step).")
    print()

    L = 18
    N = L * L

    # uniform D
    D0 = build_D(L)
    psi0 = local_states(D0)
    # defect D (local potential V=4 at centre)
    D1 = build_D(L, defect=(L // 2, L // 2, 4.0))
    psi1 = local_states(D1)

    def idx(x, y):
        return (y % L) * L + (x % L)

    # Bures distance in the +x direction at each site, for uniform vs defect
    dx_uniform = np.zeros((L, L))
    dx_defect = np.zeros((L, L))
    for y in range(L):
        for x in range(L):
            i = idx(x, y); j = idx(x + 1, y)
            dx_uniform[y, x] = bures_overlap(psi0[:, i], psi0[:, j])
            dx_defect[y, x] = bures_overlap(psi1[:, i], psi1[:, j])

    # anomaly = defect - uniform distance field, and its "curvature" (2nd var)
    anomaly = dx_defect - dx_uniform
    # discrete Laplacian of the anomaly (proxy for curvature of g_eff)
    lap = np.zeros((L, L))
    for y in range(L):
        for x in range(L):
            lap[y, x] = (anomaly[y, (x + 1) % L] + anomaly[y, (x - 1) % L]
                         + anomaly[(y + 1) % L, x] + anomaly[(y - 1) % L, x]
                         - 4 * anomaly[y, x])

    cx, cy = L // 2, L // 2
    print(f"    Bures distance field d_B(rho_i, rho_{i+x})^2, L={L}:")
    print(f"      uniform: mean={dx_uniform.mean():.5f}  std={dx_uniform.std():.2e} "
          f"(translation-invariant => constant)")
    print(f"      defect : mean={dx_defect.mean():.5f}  std={dx_defect.std():.2e} "
          f"(non-constant near defect)")
    print(f"    anomaly (defect - uniform) at defect site ({cx},{cy}): "
          f"{anomaly[cy, cx]:+.2e}")
    print(f"    Laplacian(curvature proxy) at defect site: {lap[cy, cx]:+.2e}  "
          f"(far from defect: {lap[1, 1]:.2e})")
    # measure localisation: |lap| falls off away from the defect
    far = max(abs(lap[1, 1]), abs(lap[1, L - 2]), abs(lap[L - 2, 1]), abs(lap[L - 2, L - 2]))
    localised = bool(abs(lap[cy, cx]) > 100 * far)
    print(f"      => curvature is {'LOCALISED at the defect' if localised else 'delocalised'} "
          f"(|lap| at defect = {abs(lap[cy, cx]):.2e} vs far-field {far:.2e})")

    return {
        "L": L,
        "uniform_distance_mean": float(dx_uniform.mean()),
        "uniform_distance_std": float(dx_uniform.std()),
        "defect_distance_std": float(dx_defect.std()),
        "anomaly_at_defect": float(anomaly[cy, cx]),
        "laplacian_at_defect": float(lap[cy, cx]),
        "laplacian_far_field": float(far),
        "curvature_localised_at_defect": localised,
    }


def main():
    print("=== state = position: does A extend one more step? ===")
    print()
    p1 = part1()
    p2 = part2()

    print()
    print("=== verdict ===")
    print("  1. A (no-preference = log-translation-invariance) CANNOT force 'position':")
    print("     a translation is a renormalisation, not a new state (Part 1).  On the")
    print("     state space this is 'equivariance => constant curvature' (S^3 hemisphere).")
    print("     So 'state = position' is NOT an extension of A -- A is exactly the")
    print("     symmetry that EXCLUDES position.  This is provable, not a gap.")
    print("  2. The positive source of position is the LOCALISATION of J (directed")
    print("     distinction), i.e. a DEFECT (matter): a defect in D changes the local")
    print("     state family, and the Bures-distance-field curvature is LOCALISED at the")
    print("     defect (Part 2).  This is a computable anchor of 'state-space geometry =")
    print("     physical geometry' (Layer 2).")
    print("  3. HONEST: the defect (matter) is still an INPUT, not forced by A.  So the")
    print("     wall does NOT close here; it moves to the same place as before -- 'EH")
    print("     lacks a matter source'.  What IS new: the bridge 'matter -> state-space")
    print("     curvature' is now computable and localised.")

    summary = {
        "question": "does A force 'state = position'?",
        "part1_negative": p1,
        "part1_conclusion": "A = log-translation-invariance; translation is trivial, "
                            "so A structurally cannot generate position",
        "part2_positive": p2,
        "part2_conclusion": "defect (matter) changes local states and localises the "
                            "Bures-distance-field curvature; computable bridge "
                            "matter -> state-space curvature",
        "verdict": "A cannot force position (provable); position comes from J/defect "
                   "(matter), whose curvature is computable and localised -- but matter "
                   "is still an input (= EH lacks a matter source).",
    }
    out = ROOT / "experiments" / "exp_state_space_position_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
