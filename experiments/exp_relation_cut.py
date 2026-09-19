"""Relation cut: does the localised algebra p_x D p_x (compression of D by a
position-window projection p_x) give a POSITION-DEPENDENT local spectrum
lambda(x)?  This is the "relation cut" step of the vortex path (E_x = R -> pRp,
localised relation algebra), attacking the SCALAR layer (the layer BEFORE the
"scalar -> tensor" wall).

Setup (1D ring, cleanest):
  D = nearest-neighbour hopping (+ optional local potential V at x0 as a defect)
  p_x = projection onto a window [x-m, x+m] (2m+1 sites)  <- the "relation cut"
  p_x D p_x = the window submatrix  (the localised D)
  local spectrum = eigenvalues of p_x D p_x

Question: does the local spectrum change with position?
  - uniform D  -> local spectrum translation-invariant (flat)
  - defect D   -> local spectrum changes near the defect (curvature of the SCALAR
                  field lambda(x))

HONEST BOUNDARY (up front): this computes the SCALAR field lambda(x) (position-
dependent local spectrum).  It does NOT cross the "scalar -> tensor" wall: to
turn lambda(x) into a vielbein e_mu^a(x) one needs TANGENT directions at each
point (outer derivations / infinitesimal displacements), which the discrete D
(and even the II_1 factor R, whose derivations are all INNER) does not have.
That is the SAME wall as "generate differential structure from measure".

Code: `py -m experiments.exp_relation_cut`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def ring_D(N, V=0.0, x0=None):
    D = np.zeros((N, N))
    for i in range(N):
        j = (i + 1) % N
        D[i, j] = 1.0; D[j, i] = 1.0
    if x0 is not None:
        D[x0, x0] += V
    return D


def local_spectrum(D, x, m):
    """p_x D p_x spectrum: eigenvalues of the window [x-m, x+m] submatrix."""
    N = D.shape[0]
    idx = [(x + k) % N for k in range(-m, m + 1)]
    sub = D[np.ix_(idx, idx)]
    return np.linalg.eigvalsh(sub)


def main():
    print("=== relation cut: local spectrum of p_x D p_x ===")
    print()

    N = 200
    x0 = 100
    V = 0.8
    m = 5           # window radius -> 11 sites

    D0 = ring_D(N)                 # uniform
    D1 = ring_D(N, V=V, x0=x0)     # defect

    def spectrum_profile(D):
        N = D.shape[0]
        means = np.zeros(N); stds = np.zeros(N)
        for x in range(N):
            ev = local_spectrum(D, x, m)
            means[x] = ev.mean()
            stds[x] = ev.std()
        return means, stds

    m0, s0 = spectrum_profile(D0)
    m1, s1 = spectrum_profile(D1)

    print(f"  ring N={N}, window radius m={m} (11 sites), defect V={V} at x0={x0}")
    print()
    print("  uniform ring:")
    print(f"    local spectrum mean(x):  std over x = {m0.std():.2e}  (translation-invariant)")
    print(f"    local spectrum width(x): std over x = {s0.std():.2e}  (constant)")
    print("    => local spectrum is position-INDEPENDENT (flat scalar field lambda(x)=const).")
    print()
    print("  defect ring:")
    print(f"    local spectrum mean(x):  std over x = {m1.std():.4f}  (VARIES)")
    print(f"    local spectrum width(x): std over x = {s1.std():.4f}  (VARIES)")
    print(f"    mean near defect:  {[f'{m1[(x0+d)%N]:.3f}' for d in (-3,-2,-1,0,1,2,3)]}")
    print(f"    width near defect: {[f'{s1[(x0+d)%N]:.3f}' for d in (-3,-2,-1,0,1,2,3)]}")
    print("    => local spectrum is position-DEPENDENT near the defect: a scalar field")
    print("       lambda(x) that varies with position (= the SCALAR layer, computable).")
    print()
    print("  WHERE IT STOPS (the wall):")
    print("    lambda(x) is a SCALAR (one number per site).  To get a spin-2 metric one")
    print("    needs a TENSOR e_mu^a(x) (directions at each point).  The step")
    print("    'scalar -> tensor' needs TANGENT directions = outer derivations =")
    print("    infinitesimal displacements, which the discrete D does NOT have (and the")
    print("    II_1 factor R has only INNER derivations).  That is the SAME wall as")
    print("    'generate differential structure from measure' (Connes).")

    summary = {
        "N": N, "m": m, "x0": x0, "V": V,
        "uniform_mean_std": float(m0.std()),
        "uniform_width_std": float(s0.std()),
        "defect_mean_std": float(m1.std()),
        "defect_width_std": float(s1.std()),
        "defect_mean_near": {str(d): float(m1[(x0 + d) % N]) for d in (-3, -2, -1, 0, 1, 2, 3)},
        "defect_width_near": {str(d): float(s1[(x0 + d) % N]) for d in (-3, -2, -1, 0, 1, 2, 3)},
        "conclusion": "relation cut p_x D p_x gives a computable position-dependent SCALAR "
                      "spectrum lambda(x) (flat for uniform, varying near defect).  The "
                      "scalar->tensor step is the same wall as measure->differential-structure.",
    }
    out = ROOT / "experiments" / "exp_relation_cut_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
