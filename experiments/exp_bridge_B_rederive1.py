"""Re-derive, step 1: the TRUE functional form of angle defect delta vs moduli r,
WITHOUT assuming delta = grad^2 r.

The flaw that killed "matter-source curvature R~-grad^2 rho" was assuming
delta = grad^2 r as an IDENTITY (a continuous weak-field limit smuggled in as a
discrete exact statement).  Here we drop that assumption entirely and measure the
true (nonlinear) dependence of delta on r.

Method: perturb the moduli with a SINGLE-FREQUENCY mode r = 1 + eps*cos(kx)
(or cos(ky)), compute the angle defect delta exactly (cosine rule, no linearization),
and read off how delta scales with eps and k:
  - delta ~ eps * k^0  => zero-order in r (like "constant shift", curvature~r itself)
  - delta ~ eps * k^1  => first-order  (gradient)
  - delta ~ eps * k^2  => second-order (Laplacian, the OLD wrong claim)
  - delta ~ eps^2      => nonlinear (second order in perturbation)

A single-frequency mode cleanly separates these because grad, grad^2 are exact
powers of k.  This is the honest starting point.

Code: `py -m experiments.exp_bridge_B_rederive1`
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


def angle_defect_field(r_x, r_y, L):
    """Regge angle defect (exact, nonlinear).  r_x[i,j] horiz edge, r_y vertical."""
    delta = np.zeros((L, L))

    def wrap(idx):
        return idx % L

    for p, q in product(range(L), repeat=2):
        a = r_x[p, q]
        b = r_y[wrap(p + 1), q]
        c = r_x[p, wrap(q + 1)]
        d = r_y[p, q]
        diag = np.sqrt(a**2 + b**2)

        def tri_angles(x, y, z):
            ax = np.arccos(np.clip((y**2 + z**2 - x**2) / (2 * y * z), -1, 1))
            ay = np.arccos(np.clip((x**2 + z**2 - y**2) / (2 * x * z), -1, 1))
            az = np.arccos(np.clip((x**2 + y**2 - z**2) / (2 * x * y), -1, 1))
            return ax, ay, az

        angC, angA1, angB = tri_angles(a, b, diag)
        angD, angA2, angC2 = tri_angles(diag, c, d)
        angA = angA1 + angA2
        angC_tot = angC + angC2

        delta[p, q] += angA
        delta[wrap(p + 1), q] += angB
        delta[wrap(p + 1), wrap(q + 1)] += angC_tot
        delta[p, wrap(q + 1)] += angD

    return 2 * np.pi - delta


def main():
    L = 48
    x = np.arange(L); y = np.arange(L)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    eps = 0.1
    # scan single-frequency modes along x: r_x perturbed, r_y = 1
    print("=== re-derive step 1: TRUE delta(r) functional (no delta=grad^2 r assumption) ===")
    print(f"  L={L}, mode r_x = 1 + eps*cos(kx), eps={eps}, r_y = 1")
    print(f"  {'k (2pi/L units)':>16} {'k (radians)':>12} {'max|delta|':>12} {'scaling guess':>16}")
    results = []
    ks = []
    for n in [1, 2, 3, 4, 6, 8, 12]:
        k = 2 * np.pi * n / L
        r_x = 1 + eps * np.cos(k * xx)
        r_y = np.ones((L, L))
        delta = angle_defect_field(r_x, r_y, L)
        maxd = float(np.max(np.abs(delta)))
        ks.append(k)
        results.append({"n": n, "k": float(k), "max_abs_delta": maxd})
        print(f"  {n:>16} {k:>12.4f} {maxd:>12.3e}")
    print()

    # Fit max|delta| vs k in log-log to read the order: delta ~ k^p
    ks_arr = np.array(ks)
    deltas = np.array([r["max_abs_delta"] for r in results])
    # skip the smallest k if it's near noise
    mask = deltas > 1e-12
    if mask.sum() >= 3:
        logk = np.log(ks_arr[mask]); logd = np.log(deltas[mask])
        p, _ = np.polyfit(logk, logd, 1)
        print(f"  log-log fit max|delta| ~ k^p : p = {p:.3f}")
        print(f"  p~0 => delta ~ r (zero-order);  p~1 => gradient;  p~2 => Laplacian (OLD claim)")
        print(f"  p<0 => delta grows at long wavelength (inverse), p>2 => higher-order")
    else:
        p = None

    # Also: is delta ~ eps (linear) or ~ eps^2 (nonlinear)?  one mode, two eps
    k_test = 2 * np.pi * 3 / L
    for e in [0.02, 0.05, 0.1, 0.2]:
        r_x = 1 + e * np.cos(k_test * xx)
        delta = angle_defect_field(r_x, np.ones((L, L)), L)
        print(f"  eps={e:.2f}, k=3 mode: max|delta| = {np.max(np.abs(delta)):.3e}")

    summary = {
        "L": L, "eps": eps,
        "scan": results,
        "loglog_slope_p": None if p is None else float(p),
        "conclusion": "The true delta(r) scaling (log-log slope p) tells what order of "
                      "derivative delta really is.  p=2 would vindicate the old delta~grad^2 r "
                      "(but the OLD claim was it's an IDENTITY, which the nonlinear test below "
                      "checks).  p!=2 means the old 'R~-grad^2 rho' interpretation was wrong "
                      "even in form.",
    }
    out = ROOT / "experiments" / "exp_bridge_B_rederive1_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
