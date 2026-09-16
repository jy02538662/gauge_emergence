"""Is "梯度流 × J" a genuine dynamics (spectrum changes) or just a phase rotation (spectrum invariant)?

User's sharp point: A20's "梯度流 × J" might be just a phase rotation (structure unchanged),
not real dynamics (structure changes). Key test: does D(t) = e^{iJt} D e^{-iJt} have a
time-dependent SPECTRUM?

Unitary conjugation e^{iJt} D e^{-iJt} preserves the spectrum (eigenvalues invariant) --
this is a PHASE ROTATION, not a genuine dynamics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    # static Hermitian D (2x2, e.g. a chirality-like structure)
    D = np.array([[1.0, 2.0], [2.0, -1.0]])  # Hermitian, trace 0
    # J = [[0,1],[-1,0]], anti-Hermitian (J^2 = -I), so iJ is Hermitian, e^{iJt} unitary
    J = np.array([[0.0, 1.0], [-1.0, 0.0]])

    eig0 = np.sort(np.linalg.eigvalsh(D))
    print("D (Hermitian, static) eigenvalues:", np.round(eig0, 4))
    print()
    print("D(t) = e^{iJt} D e^{-iJt} (unitary conjugation), spectrum vs t:")
    ts = [0.0, 0.5, 1.0, 1.5, 2.0]
    all_invariant = True
    for t in ts:
        U = expm(1j * J * t)
        Dt = U @ D @ U.conj().T
        eig = np.sort(np.linalg.eigvalsh(Dt))
        print(f"  t={t}: eigenvalues {np.round(eig, 4)}  (Hermitian? {np.allclose(Dt, Dt.conj().T)})")
        if not np.allclose(eig, eig0, atol=1e-9):
            all_invariant = False

    print()
    print(f"  spectrum invariant for all t? {all_invariant}")
    print()
    print("结论：")
    print("  D(t)=e^{iJt}D e^{-iJt} 的谱不随时间变（酉共轭保谱）= 相位旋转，结构不变。")
    print("  => 「梯度流 × J」是又一个「看起来像」的捷径，不是真动力学。")
    print("     真正的动力学（结构变）不是让静态 D 转相位，而是场 psi 的演化 i d_t psi = H psi。")

    out = ROOT / "experiments" / "exp_spectrum_time_invariance_last_run.json"
    out.write_text(json.dumps({
        "spectrum_invariant": bool(all_invariant),
        "eig0": eig0.tolist(),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
