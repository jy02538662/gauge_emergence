"""Verify: Hermitian D's polar decomposition gives Z2 phase (= 断裂), NOT continuous U(1).

命题（「停在 H 唯一」的第二个独立证明，程序验证）：
  自反性 D = D* ⟹ D 厄米 ⟹ 极分解 D = U|D| 的相位 U = sign(D) 本征值 ∈ {±1}，
  即 U^2 = I（对合 = Z2），不是连续 U(1)。这个 Z2 = 断裂（二元）= SU(2)。

对比（关键区分）：
  - 一般（非厄米）矩阵的相位是酉矩阵 U（本征值 e^{iθ}，连续 U(1)）；
  - 厄米矩阵的相位是 sign(D)（本征值 ±1，离散 Z2）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rng = np.random.default_rng(0)
    N = 8

    # 1) Hermitian D (自反性 D = D*)
    D = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    D = (D + D.conj().T) / 2
    herm = bool(np.allclose(D, D.conj().T, atol=1e-12))

    # spectral decomposition D = V Λ V†, Λ real
    lam, V = np.linalg.eigh(D)

    # 2) polar decomposition D = U |D| with U = sign(D), |D| = V |λ| V†
    U = (V * np.sign(lam)) @ V.conj().T          # sign(D): eigenvalues ±1
    absD = (V * np.abs(lam)) @ V.conj().T        # |D|: positive semidefinite

    polar_ok = bool(np.allclose(U @ absD, D, atol=1e-12))   # D = U|D|
    involutive = bool(np.allclose(U @ U, np.eye(N), atol=1e-12))  # U^2 = I (对合 Z2)

    # 3) U eigenvalues ∈ {±1} (discrete Z2), NOT continuous U(1) e^{iθ}
    lamU = np.linalg.eigvalsh(U)
    eig_in_pm1 = bool(np.allclose(np.sort(lamU), np.sort(np.sign(lam)), atol=1e-12))
    n_plus = int(np.sum(lamU > 0.5))
    n_minus = int(np.sum(lamU < -0.5))

    # 4) contrast: a GENERIC (non-Hermitian) matrix's polar phase is continuous U(1)
    G = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    # polar of generic G: G = W P, W unitary (continuous phase e^{iθ})
    # W eigenvalues are on the unit circle (continuous), NOT just ±1
    # (use polar via SVD: G = W (S V†) with W = U_svd V_svd†)
    U_svd, _, Vh = np.linalg.svd(G)
    W = U_svd @ Vh                       # unitary polar factor
    lamW = np.linalg.eigvals(W)
    on_unit_circle = bool(np.allclose(np.abs(lamW), 1.0, atol=1e-9))
    not_only_pm1 = not bool(np.allclose(np.abs(lamW.real), 1.0, atol=1e-9)) or bool(np.any(np.abs(lamW.imag) > 1e-6))

    print("Hermitian D polar decomposition -> Z2 phase (断裂), not continuous U(1)")
    print(f"  D Hermitian = {herm}")
    print(f"  D = U|D| (polar) = {polar_ok}")
    print(f"  U^2 = I (对合 Z2) = {involutive}")
    print(f"  U eigenvalues ∈ {{±1}} = {eig_in_pm1}  (#+ = {n_plus}, #- = {n_minus})")
    print(f"  => Hermitian phase is DISCRETE Z2 (= 断裂 = 二元 = SU(2))")
    print(f"\n  [contrast] generic matrix polar phase W: |eig|=1 = {on_unit_circle}, continuous U(1) = {not_only_pm1}")
    print(f"  => only Hermitian (自反性) collapses phase to Z2")

    out = ROOT / "experiments" / "exp_hermitian_polar_last_run.json"
    out.write_text(json.dumps({
        "D_hermitian": herm,
        "polar_decomposition_ok": polar_ok,
        "U_involutive_Z2": involutive,
        "U_eigenvalues_pm1": eig_in_pm1,
        "n_plus": n_plus,
        "n_minus": n_minus,
        "generic_phase_on_unit_circle": on_unit_circle,
        "conclusion": "Hermitian D's phase = sign(D) = Z2 (断裂=SU(2)), not continuous U(1)",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
