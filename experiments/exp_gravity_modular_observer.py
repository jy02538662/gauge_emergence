"""Modular-flow correlation G(t) = tau(x rho^{it} y rho^{-it}): which spectrum
of the observer state rho gives G(t) ~ 1/t?

Fourier core:  G(t) = int h(omega) e^{i t omega} domega
    h smooth        -> 1/t^2 or faster
    h ~ log|omega|  -> 1/t   (log singularity = scale-invariance)

For a log-uniform (scale-invariant) observer spectrum, the difference
omega = log lam_j - log lam_i has a TRIANGULAR density of states
rho(omega) = (S - |omega|)/S^2.  The operator x,y enters as a factor h(omega)
= x_ij y_ji.  So:
    G(t) = int h(omega) rho(omega) e^{i t omega} domega

Code: `py -m experiments.exp_gravity_modular_observer`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fit_power(t, G, lo=0.3, hi=0.95):
    t = np.asarray(t); G = np.abs(np.asarray(G))
    m = (t >= t.max() * lo) & (t <= t.max() * hi)
    tt, gg = t[m], G[m]
    gg = gg[gg > 1e-14]; tt = tt[:len(gg)]
    if len(tt) < 3:
        return float('nan')
    return -np.polyfit(np.log(tt), np.log(gg), 1)[0]


def G_direct(h, t, umax=8.0, n=20000):
    u = np.linspace(1e-6, umax, n)
    hp = h(u)
    return np.array([2.0 * np.trapz(hp * np.cos(tv * u), u) for tv in t])


def G_modular(h, t, S=6.0, n=20000):
    w = np.linspace(-S, S, n)
    rho = (S - np.abs(w)) / (S * S)  # triangular DOS
    hw = h(w)
    return np.array([np.trapz(hw * rho * np.cos(tv * w), w) for tv in t])


def main():
    print("=" * 74)
    print("MODULAR-FLOW CORRELATION: which observer state gives 1/t?")
    print("=" * 74)
    t = np.linspace(1.0, 120.0, 300)

    print("\n--- Part 1: Fourier core G(t) = int h(u) e^{itu} du ---")
    cases = {
        "smooth exp(-u^2)": lambda u: np.exp(-u ** 2),
        "cusp |u|": lambda u: np.abs(u) * np.exp(-u ** 2),
        "LOG  log|u|": lambda u: np.log(np.abs(u) + 1e-9) * np.exp(-u ** 2),
    }
    fourier = {}
    for name, h in cases.items():
        G = G_direct(h, t)
        p = fit_power(t, G)
        fourier[name] = p
        print(f"  {name:<20} -> tail ~ t^{{-{p:.2f}}}")

    print("\n--- Part 2: modular flow, log-uniform observer spectrum ---")
    print("  (rho(omega) = triangular DOS; operator enters as h(omega)=x_ij y_ji)")
    mod_cases = {
        "x=y=J (h=1)": lambda w: np.ones_like(w),
        "h=|omega|": lambda w: np.abs(w),
        "h=log|omega| (self-ref)": lambda w: np.log(np.abs(w) + 1e-9),
    }
    mod = {}
    for name, h in mod_cases.items():
        G = G_modular(h, t)
        p = fit_power(t, G)
        mod[name] = p
        print(f"  {name:<26} -> tail ~ t^{{-{p:.2f}}}")

    out = ROOT / "experiments" / "exp_gravity_modular_observer_last_run.json"
    out.write_text(json.dumps({"fourier": fourier, "modular": mod}, indent=2),
                   encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
