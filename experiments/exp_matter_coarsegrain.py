"""Coarse-graining: does averaging the long-range Friedel oscillation of the
bond-order stress leave a LOCAL effective source?

Correction (per user): "conservation law fails" was WRONG wording.  The
conservation law d_j T^ij = f^i is a local differential equation (Noether); it
does not "fail" -- what is true is that T^ij ITSELF is long-range (Friedel
oscillation) in the gapless system.  In GR the SOURCE T_mu nu is local and the
long-range is in the SOLUTION g_mu nu ~ 1/r.  So the task is not "fix the
conservation law" but "coarse-grain the microscopic source to a local effective
source", exactly as screening does in condensed matter.

This probe tests the coarse-graining step:
  1. bare bond-order stress T_{i,i+1} = 2 Re K_{i,i+1}
  2. bare divergence g_i = T_{i,i+1} - T_{i-1,i}  (long-range Friedel, wavelength 2)
  3. coarse-grain: G_i = (1/(2m+1)) sum_{k=-m}^{m} g_{i+k}   (window average)
  4. localisation ratio = max|G near defect| / max|G far field|  as function of m

Criterion: if G_i localises at the defect as m grows (ratio -> large), then
coarse-graining turns the long-range Friedel source into a LOCAL effective source
-- the "last hop" of the matter source, without touching the curvature wall.

Code: `py -m experiments.exp_matter_coarsegrain`
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


def bond_stress_divergence(N, x0, V):
    D = ring_D(N, V=V, x0=x0)
    ev, Vm = np.linalg.eigh(D)
    occ = ev < 0.0
    P = Vm[:, occ] @ Vm[:, occ].conj().T
    sig = np.array([2.0 * np.real(P[i, (i + 1) % N]) for i in range(N)])
    g = sig - np.roll(sig, 1)         # discrete divergence
    return g


def localisation(g, x0):
    near = np.max(np.abs(g[(x0 - 4) % len(g):(x0 + 5) % len(g)]))
    far = max(np.max(np.abs(g[0:len(g) // 4])),
              np.max(np.abs(g[3 * len(g) // 4:])))
    return near, far


def main():
    print("=== coarse-graining: long-range Friedel source -> local effective source ===")
    print()

    N = 120
    x0 = 60
    V = 0.8

    g = bond_stress_divergence(N, x0, V)

    print(f"  ring N={N}, defect V={V} at x0={x0} (gapless, half-filling)")
    print()
    print("  bare divergence g_i (m=0):")
    near, far = localisation(g, x0)
    print(f"    |g| near defect = {near:.4f}   far field = {far:.4f}   ratio = {near/max(far,1e-12):.1f}")
    print(f"    (long-range Friedel oscillation, wavelength ~2: the STRESS is long-range,")
    print(f"     not 'conservation fails')")
    print()
    print("  coarse-grained G_i = window-average(g, m):")
    rows = []
    for m in (0, 1, 2, 3, 4, 6, 8):
        # circular window average
        G = np.zeros(N)
        for i in range(N):
            s = 0.0
            for k in range(-m, m + 1):
                s += g[(i + k) % N]
            G[i] = s / (2 * m + 1)
        nearG, farG = localisation(G, x0)
        ratio = nearG / max(farG, 1e-12)
        rows.append((m, nearG, farG, ratio))
        print(f"    m={m}: |G| near defect = {nearG:.4f}   far field = {farG:.4f}   "
              f"ratio = {ratio:.1f}")

    # does ratio grow (localisation improve) with m?
    ratios = [r[3] for r in rows]
    improves = bool(ratios[-1] > 20 * ratios[0] and ratios[-1] > 100)
    print()
    print("  conclusion:")
    if improves:
        print(f"    => coarse-graining DOES localise: ratio {ratios[0]:.1f} -> {ratios[-1]:.1f}.")
        print("       Averaging the Friedel oscillation (wavelength 2) leaves a LOCAL")
        print("       effective source.  This is the 'screening' step: microscopic source")
        print("       long-range -> coarse-grained effective source local.")
    else:
        print(f"    => coarse-graining does NOT localise (ratio {ratios[0]:.1f} -> {ratios[-1]:.1f}).")
        print("       The long-range part is not just an oscillating tail; needs RPA screening")
        print("       (an interaction) or a different stress definition.")
    print("  NOTE: this does NOT touch the curvature wall; it is the matter-side")
    print("        'source vs solution' separation, standard in condensed matter.")

    summary = {
        "N": N, "x0": x0, "V": V,
        "bare_ratio": float(ratios[0]),
        "coarsegrain_rows": [{"m": r[0], "near": r[1], "far": r[2], "ratio": r[3]} for r in rows],
        "localisation_improves_with_m": improves,
        "conclusion": "coarse-graining averages the Friedel oscillation; whether it leaves "
                      "a local effective source is reported above",
    }
    out = ROOT / "experiments" / "exp_matter_coarsegrain_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
