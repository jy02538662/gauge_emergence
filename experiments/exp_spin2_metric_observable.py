"""Step 4b: which observable is "self-referential" (h(omega) ~ log|omega|)?

Step 4 (exp_spin2_metric_observer) showed: 1/t comes from the OBSERVABLE's log
singularity (h ~ log|omega|), NOT the observer state rho.  The wall: the metric
observable must ITSELF be scale-invariant (self-referential).  Per 核查与建议,
check candidate 1: D^2's spectral weight h(omega) under the self-referential rho.

For an observable x (in the same basis as rho), the modular-frequency weight is
  h(omega) = sum_{theta_ij ~ omega} rho_i x_ij x_ji,   theta_ij = ln(lambda_i/lambda_j).
Log singularity h ~ log|omega| -> G(t) ~ 1/t (long/continuous); delta/smooth -> short.

Check D^2 (candidate 1) in TWO bases, vs deltaF (step-4, known smooth) and J
(modular conjugation, the known self-referential observable).

Code: `py -m experiments.exp_spin2_metric_observable`
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


def mag_trans(D, L, axis):
    N = L ** 3
    T = np.zeros((N, N))
    for x, y, z in product(range(L), repeat=3):
        i = idx(x, y, z, L)
        if axis == 'x':
            j = idx(x + 1, y, z, L)
        elif axis == 'y':
            j = idx(x, y + 1, z, L)
        else:
            j = idx(x, y, z + 1, L)
        T[i, j] = D[i, j]
    return T


def kramers_J(D):
    """Kramers J (J^2=-I, JD=DJ) for all-even spectrum."""
    n = D.shape[0]
    ev, V = np.linalg.eigh(D)
    J = np.zeros((n, n))
    used = np.zeros(n, bool)
    for i in range(n):
        if used[i]:
            continue
        for j in range(i + 1, n):
            if not used[j] and abs(ev[i] - ev[j]) < 1e-9:
                a, b = V[:, i], V[:, j]
                J += np.outer(a, b) - np.outer(b, a)
                used[i] = used[j] = True
                break
        else:
            used[i] = True
    return J


def h_histogram(x, rho, bins=121, S=6.0):
    """h(omega) = sum_{theta~omega} rho_i x_ij x_ji, theta = ln(lambda_i/lambda_j)."""
    lr = np.log(rho)
    theta = lr[:, None] - lr[None, :]
    w = rho[:, None] * (x * x.T)            # rho_i (x)_ij (x)_ji
    bins_edges = np.linspace(-S, S, bins)
    hist = np.zeros(bins - 1)
    for b in range(bins - 1):
        m = (theta >= bins_edges[b]) & (theta < bins_edges[b + 1])
        hist[b] = float(np.sum(np.abs(w[m])))
    centers = 0.5 * (bins_edges[1:] + bins_edges[:-1])
    return centers, hist


def main():
    L = 4
    N = L ** 3
    print("=== step 4b: which observable is self-referential (h ~ log|omega|)? ===")
    print()

    D = torus3D(L, True)
    D2 = D @ D
    ev, U = np.linalg.eigh(D)
    D2_E = U.T @ D2 @ U                         # D^2 in energy basis (diagonal)
    D2_pos = D2                                 # D^2 in position basis (hopping)
    J = kramers_J(D)                            # modular conjugation (self-referential)

    # defect curvature deltaF (step 4, known smooth)
    D1 = torus3D(L, True)
    D1[idx(0, 0, 0, L), idx(1, 0, 0, L)] *= -1
    D1[idx(1, 0, 0, L), idx(0, 0, 0, L)] *= -1
    F0 = mag_trans(D, L, 'x') @ mag_trans(D, L, 'y') - mag_trans(D, L, 'y') @ mag_trans(D, L, 'x')
    F1 = mag_trans(D1, L, 'x') @ mag_trans(D1, L, 'y') - mag_trans(D1, L, 'y') @ mag_trans(D1, L, 'x')
    dF = F1 - F0
    dF_E = U.T @ dF @ U

    # self-referential rho (log-uniform)
    s = 6.0 * np.arange(N) / (N - 1)
    rho = np.exp(s); rho = rho / rho.sum()

    def near0_report(name, x):
        centers, hist = h_histogram(x, rho)
        near0 = hist[np.abs(centers) < 0.3]
        peak_idx = int(np.argmax(hist))
        # log-singularity test: does h(omega) GROW as omega -> 0?
        left = hist[np.abs(centers) < 0.6]
        left_centers = centers[np.abs(centers) < 0.6]
        # slope of log(h) vs log(|omega|) near 0 (negative slope ~ log singularity)
        keep = (left > 1e-16) & (np.abs(left_centers) > 1e-3)
        slope = float('nan')
        if keep.sum() >= 3:
            slope = float(np.polyfit(np.log(np.abs(left_centers[keep])),
                                     np.log(left[keep]), 1)[0])
        print(f"  {name}: total weight near |omega|<0.3 = {near0.sum():.3e} "
              f"(of {hist.sum():.3e}), peak at omega={centers[peak_idx]:+.2f}, "
              f"slope of log h vs log|omega| = {slope:.3f}")
        return slope

    print("h(omega) of candidate observables (self-referential rho, energy basis):")
    s_D2_E = near0_report("D^2 (energy basis, diagonal)", D2_E)
    s_D2_pos = near0_report("D^2 (position basis, hopping)", D2_pos)
    s_dF = near0_report("deltaF (defect, step 4)", dF_E)
    s_J = near0_report("J (modular conjugation)", U.T @ J @ U)

    print()
    print("  log singularity -> slope of log h vs log|omega| is NEGATIVE (h diverges at 0).")
    print("  delta/smooth    -> h concentrated at single omega (peak, no divergence).")

    summary = {
        "slope_D2_energy": s_D2_E,
        "slope_D2_position": s_D2_pos,
        "slope_deltaF": s_dF,
        "slope_J": s_J,
        "candidate1_D2_has_log_singularity": bool(s_D2_E < -0.5) or bool(s_D2_pos < -0.5),
        "note": "which observable carries h~log|omega| (self-referential metric candidate).",
    }
    out = ROOT / "experiments" / "exp_spin2_metric_observable_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
