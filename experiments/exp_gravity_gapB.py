"""Gap B verification: is the 1/r from the modular-flow route the GEOMETRIC
(Dirac) propagator, or an arbitrary correlation?

The geometric Green's function of the pi-flux Dirac operator D is
    G = (D + i eps)^{-1}
At the Dirac points (E=0), the massless 2D Dirac propagator decays as 1/r.
If |G(r)| ~ 1/r on the lattice, then the "1/r" IS geometric (the Dirac
propagator), closing gap B (1/r is a geometric potential, not arbitrary).

Code: `py -m experiments.exp_gravity_gapB`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def toroidal_D(n_per_dim, pi_flux=True):
    n = n_per_dim ** 2
    D = np.zeros((n, n), complex)
    for i in range(n_per_dim):
        for j in range(n_per_dim):
            idx = n_per_dim * i + j
            jr = (j + 1) % n_per_dim
            D[idx, n_per_dim * i + jr] += 1.0
            D[n_per_dim * i + jr, idx] += 1.0
            ph = np.pi * j if pi_flux else 0.0
            D[idx, n_per_dim * ((i + 1) % n_per_dim) + j] += np.exp(1j * ph)
            D[n_per_dim * ((i + 1) % n_per_dim) + j, idx] += np.exp(-1j * ph)
    return D


def graph_dist(npd, a, b):
    ai, aj = divmod(a, npd); bi, bj = divmod(b, npd)
    return min(abs(ai - bi), npd - abs(ai - bi)) + min(abs(aj - bj), npd - abs(aj - bj))


def main():
    print("=" * 74)
    print("GAP B: geometric (Dirac) propagator on the pi-flux lattice")
    print("=" * 74)

    npd = 20
    n = npd ** 2
    D = toroidal_D(npd, pi_flux=True)
    lam = np.linalg.eigvalsh(D)
    print(f"N={n} ({npd}x{npd}), D spectrum [{lam[0]:.3f}, {lam[-1]:.3f}] "
          f"(Dirac points at E=0)")

    # Dirac propagator with small regularization
    eps = 0.05
    G = np.linalg.inv(D + 1j * eps * np.eye(n))

    # average |G_ij| vs graph distance from node 0
    dists = {}
    for j in range(n):
        d = graph_dist(npd, 0, j)
        dists.setdefault(d, []).append(abs(G[0, j]))

    print(f"\n  {'dist':>5} {'|G(r)|':>12} {'ratio':>8}")
    ds = sorted(dists.keys())
    vals = [float(np.mean(dists[d])) for d in ds]
    prev = None
    for d, v in zip(ds, vals):
        ratio = f"{v/prev:.2f}" if prev else "—"
        print(f"  {d:>5} {v:>12.5e} {ratio:>8}")
        prev = v

    # fit power law on the mid-range (avoid r=0 and the boundary)
    ds_arr = np.array(ds, float)
    vals_arr = np.array(vals)
    m = (ds_arr >= 2) & (ds_arr <= npd)  # mid range
    p = -np.polyfit(np.log(ds_arr[m]), np.log(vals_arr[m]), 1)[0]
    print(f"\n  fit |G(r)| ~ r^{{-{p:.3f}}}  (2D massless Dirac => p=1;  scalar => log r;  gapped => exponential)")

    verdict = ("GEOMETRIC 1/r (Dirac propagator)" if 0.7 < p < 1.4
               else ("log r (massless scalar)" if p < 0.3 else "other (not 1/r)"))
    print(f"  VERDICT: {verdict}")

    out = ROOT / "experiments" / "exp_gravity_gapB_last_run.json"
    out.write_text(json.dumps({"power": float(p), "dists": ds, "vals": vals,
                               "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
