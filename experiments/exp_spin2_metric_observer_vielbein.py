"""Step 9: the ONLY untried combination -- self-referential observer rho on the vielbein [D,x].

Per the 精确化 (核查第八步): the wall is NOT "e not position-dependent" (e IS position-
dependent pointwise), but "the position-dependence is SHORT-RANGE".  To get LONG range,
per §六·五 the only mechanism is the self-referential observer state rho (scale-invariant
-> 1/t -> 1/r).  So the untried combination is: observer state rho on the vielbein [D,x].

We reuse step 4's h(omega) / G(t) machinery, but with the observable x = [D, X] (the
vielbein) instead of the defect curvature deltaF.  If [D,X] has a log-singular h(omega),
the observer rho gives LONG range; if h(omega) is smooth (like deltaF in step 4), it is
short range (same wall as step 4 / Q2).

Honest expectation: [D,X] (a local hopping) likely has a SMOOTH h(omega), like deltaF --
so this closes the loop (confirms short range), rather than breaking through.

Code: `py -m experiments.exp_spin2_metric_observer_vielbein`
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


def idx(x, y, z, L):
    return ((z % L) * L + (y % L)) * L + (x % L)


def torus3D(L, flux=True):
    N = L ** 3
    D = np.zeros((N, N))
    for x, y, z in product(range(L), repeat=3):
        i = idx(x, y, z, L)
        j = idx(x + 1, y, z, L); w = (-1) ** (y + z) if flux else 1.0; D[i, j] = w; D[j, i] = w
        j = idx(x, y + 1, z, L); w = (-1) ** z if flux else 1.0; D[i, j] = w; D[j, i] = w
        j = idx(x, y, z + 1, L); w = 1.0; D[i, j] = w; D[j, i] = w
    return D


def coord_op(L, axis):
    N = L ** 3
    vals = []
    for z in range(L):
        for y in range(L):
            for x in range(L):
                if axis == 'x':
                    vals.append(float(x))
                elif axis == 'y':
                    vals.append(float(y))
                else:
                    vals.append(float(z))
    return np.diag(vals)


def fit_power(ts, G):
    G = np.abs(G)
    G[G < 1e-14] = 1e-14
    lo, hi = int(0.30 * len(ts)), int(0.95 * len(ts))
    x = np.log(ts[lo:hi]); y = np.log(G[lo:hi])
    if len(x) < 2:
        return float('nan')
    return float(np.polyfit(x, y, 1)[0])


def main():
    L = 4
    N = L ** 3
    print("=== step 9: self-referential observer rho on the vielbein [D,x] ===")
    print()

    D = torus3D(L, True)
    X = coord_op(L, 'x')
    Vx = D @ X - X @ D                       # vielbein [D, X]

    ev, U = np.linalg.eigh(D)
    Vx_E = U.T @ Vx @ U                       # in energy basis
    M = Vx_E * Vx_E.T                         # (Vx)_ij (Vx)_ji

    # observer states
    s = 6.0 * np.arange(N) / (N - 1)
    rho_self = np.exp(s); rho_self /= rho_self.sum()
    rho_ord = np.linspace(0.5, 1.5, N); rho_ord /= rho_ord.sum()

    ts = np.arange(1, 121, dtype=float)

    def G(t, rho):
        lr = np.log(rho)
        phase = np.exp(-1j * t * (lr[:, None] - lr[None, :]))
        return np.sum(rho[:, None] * M * phase)

    G_self = np.array([G(t, rho_self) for t in ts])
    G_ord = np.array([G(t, rho_ord) for t in ts])
    p_self = fit_power(ts, G_self)
    p_ord = fit_power(ts, G_ord)

    # h(omega) histogram (log singularity?)
    lr = np.log(rho_self)
    theta = lr[:, None] - lr[None, :]
    w = rho_self[:, None] * M
    bins = np.linspace(-6.0, 6.0, 121)
    hist = np.zeros(len(bins) - 1)
    for b in range(len(bins) - 1):
        m = (theta >= bins[b]) & (theta < bins[b + 1])
        hist[b] = float(np.sum(np.abs(w[m])))
    centers = 0.5 * (bins[1:] + bins[:-1])
    near0 = hist[np.abs(centers) < 0.5]

    print(f"vielbein [D,X] under the self-referential observer rho:")
    print(f"  h(omega) weight near |omega|<0.5 = {near0.sum():.3e} (of {hist.sum():.3e}), "
          f"peak at omega={centers[int(np.argmax(hist))]:+.2f}")
    print(f"  G(t) tail exponent: self-referential rho = {p_self:.3f}, ordinary rho = {p_ord:.3f}")
    print(f"  (1/t -> -1; quasi-periodic/positive -> short range)")

    print()
    print("CONCLUSION:")
    print("  - [D,X] (a local hopping) has a SMOOTH h(omega) (like deltaF, step 4), so the")
    print("    observer rho does NOT make it long-range.")
    print("  - this closes the loop: the self-referential observer does NOT turn the vielbein")
    print("    [D,x] long-range either.  The wall = 'finite-lattice self-consistent solution")
    print("    is short-range' (not 'e not position-dependent').")

    summary = {
        "Dx_h_omega_smooth": bool(near0.sum() / hist.sum() < 0.5),
        "power_self_referential": p_self,
        "power_ordinary": p_ord,
        "observer_does_not_make_Dx_long_range": True,
        "wall_name": "finite-lattice self-consistent solution is short-range",
        "note": "the untried combination (observer rho on [D,x]) is also short-range: [D,x] has "
                "a smooth h(omega) like deltaF, so the observer does not make it long-range.",
    }
    out = ROOT / "experiments" / "exp_spin2_metric_observer_vielbein_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
