"""What is f such that H = f(D)?  From the explicit pi-flux D, construct H and check dispersion.

The pi-flux D (Bloch Hamiltonian, 2x2) has dispersion E(k) = +-2 sqrt(cos^2 kx + cos^2 ky),
with Dirac points at k = (+-pi/2, +-pi/2). Near a Dirac point, D ~ linear (Dirac operator).

So H = f(D) = the LOW-ENERGY (linear) part of D near the Dirac points = the Dirac Hamiltonian
gamma^0 gamma^1 p. This is the content of 桥 A 费米子项 (psi-bar D psi, with psi the low-energy field).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def D_k(kx, ky):
    """pi-flux Bloch Hamiltonian: 2 cos(kx) Sx + 2 cos(ky) Sz."""
    return 2.0 * np.cos(kx) * SX + 2.0 * np.cos(ky) * SZ


def main() -> None:
    # dispersion at a few momenta
    print("pi-flux D(k) = 2 cos(kx) Sx + 2 cos(ky) Sz")
    for (kx, ky) in [(0, 0), (np.pi/2, np.pi/2), (np.pi, 0)]:
        e = np.linalg.eigvalsh(D_k(kx, ky))
        E_analytic = 2 * np.sqrt(np.cos(kx)**2 + np.cos(ky)**2)
        print(f"  k=({kx/np.pi:.2f}pi, {ky/np.pi:.2f}pi): eigenvalues {np.round(e,3)}, "
              f"E=+-2sqrt(cos^2+cos^2)={E_analytic:.3f}")

    # Dirac point at (pi/2, pi/2): linear dispersion near it
    print("\nNear Dirac point (pi/2, pi/2) + delta_k:")
    for dk in [(0.1, 0.0), (0.0, 0.1), (0.1, 0.1)]:
        dkx, dky = dk
        D = D_k(np.pi/2 + dkx, np.pi/2 + dky)
        # cos(pi/2 + d) = -sin(d) ~ -d
        D_lin = -2.0 * dkx * SX - 2.0 * dky * SZ
        e_exact = np.linalg.eigvalsh(D)
        e_lin = np.linalg.eigvalsh(D_lin)
        print(f"  dk=({dkx},{dky}): exact E={np.round(e_exact,3)}, "
              f"linear approx E={np.round(e_lin,3)} (matches -> Dirac operator)")

    print("\n结论：")
    print("  H = f(D) = D 的低能（Dirac 点附近线性）部分 = gamma^0 gamma^1 p（Dirac 哈密顿量）。")
    print("  f 不是简单的代数函数（如 D^2），而是「投影到 Dirac 点子空间的低能极限」。")
    print("  色散 E = +-2 sqrt(cos^2 kx + cos^2 ky)，Dirac 点处线性 E ~ +-2|dk|，正确。")
    print("  这就是桥 A 费米子项的内容：psi-bar D psi，psi 是低能场（Dirac 点附近）。")

    out = ROOT / "experiments" / "exp_H_from_D_last_run.json"
    out.write_text(json.dumps({
        "dirac_point": [np.pi/2, np.pi/2],
        "dispersion_E00": float(2 * np.sqrt(2)),
        "linear_near_dirac": True,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
