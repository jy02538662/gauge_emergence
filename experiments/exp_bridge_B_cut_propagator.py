"""Candidate 3 check: does CUTTING create 1/r, or is 1/r already there (Dirac points)?

Candidate 3 (user): "the defect does not live in space; the defect CREATES space.
Observation (cutting somewhere) -> local space appears -> 1/r is the propagator on
that space."

HONEST check (avoiding a circularity): the check "propagator after cutting gives
1/r" is circular (r = lattice distance presupposes space) AND confounded (Dirac
points already give 1/r).  So the real question is: does CUTTING CHANGE the
propagator at all?

Compare:
  G(r)  = |(D^2 + eps^2 I)^{-1}|[0 -> r]   (uniform pi-flux, no cut)
  G'(r) = |(D'^2 + eps^2 I)^{-1}|[c -> r]  (cut: sever all edges of center vertex c)

If G' == G (both 1/r, unchanged): cutting did NOT create 1/r -- it was already
there from the Dirac points (gapless massless modes).  Candidate 3's "cut creates
1/r" is then refuted (the LONG-range is a Dirac-point property, not a cut property).
If G' has a NEW structure near the cut absent from G: cutting did introduce
something.

Code: `py -m experiments.exp_bridge_B_cut_propagator`
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


def torus2D(L, flux=True):
    N = L * L

    def idx(x, y):
        return (y % L) * L + (x % L)

    D = np.zeros((N, N))
    for x, y in product(range(L), repeat=2):
        i = idx(x, y)
        j = idx(x + 1, y); w = (-1) ** y if flux else 1.0; D[i, j] = w; D[j, i] = w
        j = idx(x, y + 1); w = 1.0; D[i, j] = w; D[j, i] = w
    return D


def graph_dist(L, a, b):
    ai, aj = divmod(a, L)
    bi, bj = divmod(b, L)
    di = min(abs(ai - bi), L - abs(ai - bi))
    dj = min(abs(aj - bj), L - abs(aj - bj))
    return di + dj


def propagator(D, eps, src):
    """|G(src -> j)| = |(D^2 + eps^2 I)^{-1}[src, j]| as a function of graph distance."""
    N = D.shape[0]
    inv = np.linalg.inv(D @ D + eps**2 * np.eye(N))
    return np.abs(inv[src, :])


def radial_profile(G, L, src, rmax):
    out = []
    for r in range(1, rmax + 1):
        vals = [G[j] for j in range(L * L) if graph_dist(L, src, j) == r]
        if vals:
            out.append((r, float(np.mean(vals))))
    return out


def main():
    L = 48
    eps = 0.2
    N = L * L
    src = 0                      # origin for uniform
    center = (L // 2) * L + (L // 2)   # center vertex to cut

    D = torus2D(L, flux=True)

    # uniform propagator (no cut)
    Gu = propagator(D, eps, src)
    prof_u = radial_profile(Gu, L, src, 16)

    # cut: sever all edges of the center vertex (keep diagonal = 0)
    Dc = D.copy()
    Dc[center, :] = 0.0
    Dc[:, center] = 0.0
    Gc = propagator(Dc, eps, src)   # from a vertex FAR from the cut (src=0)
    prof_c_far = radial_profile(Gc, L, src, 16)

    # also from the cut itself (a neighbor of center, since center is isolated)
    # pick a neighbor of center
    nbr = None
    for j in range(N):
        if D[center, j] != 0:
            nbr = j
            break
    Gc2 = propagator(Dc, eps, nbr)
    prof_c_near = radial_profile(Gc2, L, nbr, 16)

    print("=== does cutting change the propagator? (uniform vs cut) ===")
    print(f"  L={L}, eps={eps}")
    print(f"  {'r':>4} {'uniform G(r)':>14} {'cut(far) G(r)':>14} {'cut(near) G(r)':>14}")
    for r in range(1, 13):
        u = prof_u[r-1][1] if r-1 < len(prof_u) else 0
        cf = prof_c_far[r-1][1] if r-1 < len(prof_c_far) else 0
        cn = prof_c_near[r-1][1] if r-1 < len(prof_c_near) else 0
        print(f"  {r:>4} {u:>14.4e} {cf:>14.4e} {cn:>14.4e}")

    # fit uniform: G ~ r^-p
    rs = np.array([prof_u[k][0] for k in range(3, 12)])
    gs = np.array([prof_u[k][1] for k in range(3, 12)])
    p = -np.polyfit(np.log(rs), np.log(gs), 1)[0]
    print(f"\n  uniform G(r) ~ r^-p, p = {p:.3f}  (Dirac points give ~1/r)")
    print(f"  => cut(far) vs uniform: difference shows whether cutting changed anything.")

    summary = {
        "L": L, "eps": eps,
        "uniform_p": float(p),
        "uniform_G": {str(prof_u[k][0]): prof_u[k][1] for k in range(6)},
        "cut_far_G": {str(prof_c_far[k][0]): prof_c_far[k][1] for k in range(6)},
        "cut_near_G": {str(prof_c_near[k][0]): prof_c_near[k][1] for k in range(6)},
        "conclusion": "If cut(far) ~ uniform (both 1/r), cutting did NOT create 1/r -- "
                      "it was already there from Dirac points. Then candidate 3's 'cut "
                      "creates 1/r' is refuted; the long-range is a Dirac-point property.",
    }
    out = ROOT / "experiments" / "exp_bridge_B_cut_propagator_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
