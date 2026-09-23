"""Does Cl(3) (from 3D pi-flux anti-commuting magnetic translations) carry SU(3)?

命题（选项 C 查证）：3D pi-flux 的 T_x,T_y,T_z（三对反对易）生成 Cl(3)，Cl(3) 有 8 维
基。这 8 维能不能承载 su(3) 的 8 个生成元（含 λ_4567 那 4 个混合生成元）？

验证：对比「Cl(3) 的 Lie 代数（交换子）实维数」vs「su(3) 的 Lie 代数实维数」。
Cl(3) ≅ M_2(C)，交换子张成 sl(2,C)（6 实维，紧形式 su(2)=3 维）；su(3) 是 8 维。
「8 维基」≠「8 维 Lie 代数」——前者是代数维数（含中心 I、iI），后者才是生成元计数。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Pauli matrices
s1 = np.array([[0, 1], [1, 0]], dtype=complex)
s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
s3 = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def comm(A, B):
    return A @ B - B @ A


def real_rank(matrices, tol=1e-8):
    vecs = np.array([np.concatenate([M.real.ravel(), M.imag.ravel()]) for M in matrices])
    return int(np.linalg.matrix_rank(vecs, tol=tol))


def main() -> None:
    # Cl(3) generators e_i = i sigma_i (e_i^2 = -1, anti-commute)
    e1, e2, e3 = 1j * s1, 1j * s2, 1j * s3
    # 8 real basis of Cl(3) = M_2(C): I, iσ1, iσ2, iσ3, σ1, σ2, σ3, iI
    cl3_basis = [I2, e1, e2, e3, s1, s2, s3, 1j * I2]
    cl3_comms = [comm(A, B) for A in cl3_basis for B in cl3_basis]
    cl3_rank = real_rank(cl3_comms)

    # su(3): 8 Gell-Mann matrices
    gm = [None] * 9
    gm[1] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    gm[2] = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex)
    gm[3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex)
    gm[4] = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex)
    gm[5] = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex)
    gm[6] = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex)
    gm[7] = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex)
    gm[8] = (1 / np.sqrt(3)) * np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex)
    su3_basis = [gm[i] for i in range(1, 9)]
    su3_comms = [comm(A, B) for A in su3_basis for B in su3_basis]
    su3_rank = real_rank(su3_comms)

    print("Cl(3) basis = 8 real dims (as algebra M_2(C))")
    print(f"  commutator Lie algebra real dim = {cl3_rank}  (expected 6 = sl(2,C); compact su(2)=3)")
    print(f"su(3) Gell-Mann Lie algebra real dim = {su3_rank}  (expected 8)")
    print()
    print("=> Cl(3) carries sl(2,C)/su(2), NOT su(3).")
    print("   '8 basis dims = 8 generators' is a DIMENSION COINCIDENCE (数出 8 的老坑).")
    print("   3D pi-flux (Cl(3)) cannot carry λ_4567 (the 4 mixing generators of SU(3)).")

    out = ROOT / "experiments" / "exp_clifford_su3_last_run.json"
    out.write_text(json.dumps({
        "cl3_lie_real_rank": cl3_rank,
        "su3_lie_real_rank": su3_rank,
        "conclusion": "Cl(3) Lie algebra (sl(2,C)/su(2)) != su(3); 8=8 is a dimension coincidence",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
