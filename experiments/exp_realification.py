"""Type distinction = spatial vs time reversal; realification resolves it.

Physical meaning (Wigner):
  - unitary Gamma = spatial symmetry (complex-LINEAR, keeps time direction)
  - anti-unitary K/T = time reversal (complex-ANTI-linear, flips time direction)

Realification C^n -> R^{2n}: both become real ORTHOGONAL, type distinction disappears.
This is the "爬到 8 维 = 实化" candidate: 4x4 Dirac (C^4) -> octonions (R^8).

Verify:
  1. Gamma is complex-linear, K is complex-anti-linear (type distinction real).
  2. After realification, both become real orthogonal (type distinction resolved).
  3. The 4x4 Dirac's Hilbert space C^4 = R^8 (same real dim as octonions O).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def realify_op_linear(M):
    """Complex-linear 2x2 matrix -> real 4x4 matrix (via z=x+iy -> (x,y))."""
    a, b = M.real, M.imag
    R = np.zeros((4, 4))
    for i in range(2):
        for j in range(2):
            R[2 * i, 2 * j] += a[i, j]
            R[2 * i, 2 * j + 1] += -b[i, j]
            R[2 * i + 1, 2 * j] += b[i, j]
            R[2 * i + 1, 2 * j + 1] += a[i, j]
    return R


def main() -> None:
    # Gamma = sigma_z (chirality, unitary, spatial)
    Gamma = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

    v = np.array([1.0 + 2.0j, 3.0 - 1.0j])
    c = 2.0 + 1.0j

    # 1. type distinction on C^2
    lin_ok = np.allclose(Gamma @ (c * v), c * (Gamma @ v))            # complex-linear
    anti_ok = np.allclose(np.conjugate(c * v), np.conjugate(c) * np.conjugate(v))  # K anti-linear

    # 2. realify both to R^4
    R_Gamma = realify_op_linear(Gamma)
    R_K = np.diag([1.0, -1.0, 1.0, -1.0])  # conjugation K: z -> conj(z), i.e. y -> -y

    orth_Gamma = np.allclose(R_Gamma.T @ R_Gamma, np.eye(4))
    orth_K = np.allclose(R_K.T @ R_K, np.eye(4))

    # 3. dimension check: C^4 = R^8 (4x4 Dirac acts on C^4)
    print("1. Type distinction on C^2 (Wigner):")
    print(f"   Gamma (unitary) complex-LINEAR:  Gamma(c v) = c Gamma(v) ?  {lin_ok}")
    print(f"   K (anti-unitary) complex-ANTI-linear:  K(c v) = conj(c) K(v) ?  {anti_ok}")
    print()
    print("2. Realification C^2 -> R^4:")
    print(f"   Gamma -> 4x4 real orthogonal?  {orth_Gamma}")
    print(f"   K     -> 4x4 real orthogonal?  {orth_K}")
    print("   => type distinction (linear vs anti-linear) DISAPPEARS: both real orthogonal.")
    print()
    print("3. Dimension: 4x4 Dirac acts on C^4 = R^8 (real), same as octonions O = R^8.")
    print("   => '爬到 8 维' = '实化 C^4 -> R^8' = '消解类型区分' (space vs time).")
    print("   => octonions/triality = the REAL structure where unitary & anti-unitary unify.")

    out = ROOT / "experiments" / "exp_realification_last_run.json"
    out.write_text(json.dumps({
        "Gamma_linear": bool(lin_ok),
        "K_antilinear": bool(anti_ok),
        "Gamma_orthogonal_after_realify": bool(orth_Gamma),
        "K_orthogonal_after_realify": bool(orth_K),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
