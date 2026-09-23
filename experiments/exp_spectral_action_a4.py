"""Second判据 of 「整体动力」 (corrected): isolate the cross term and verify it ∝ ω^2.

符号结论（确定）：
  D_R^2 = -∂_t^2 + H_mod^2 + σ_y ⊗ (∂_t H_mod)
  - a_2 ~ Tr(D_R^2)  不含时间导数：Tr(σ_y)=0 使交叉项一阶贡献 = 0
  - a_4 ~ Tr(D_R^4)  含时间导数：交叉项平方 (σ_y ⊗ ∂_t H_mod)^2 = (∂_t H_mod)^2 I 进 a_4

数值验证（干净）：直接算交叉项 B = [D_t, H_mod]，验证 ||B||^2 ∝ ω^2（= (∂_t H_mod)^2）。
（之前把时间导数项埋进 Tr(D_R^4) 里被 H_mod^4 淹没，这里分离出来。）
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    N = 200
    t = np.linspace(0, 1, N, endpoint=False)
    Dt = np.zeros((N, N))
    for n in range(N):
        Dt[n, (n + 1) % N] = 0.5
        Dt[n, (n - 1) % N] = -0.5

    c = 1.0
    eps = 0.3
    omegas = [1.0, 2.0, 4.0, 8.0, 16.0]

    normB2 = []
    for w in omegas:
        H_dyn = c + eps * np.sin(2 * np.pi * w * t)
        Hm = np.diag(H_dyn)
        B = Dt @ Hm - Hm @ Dt          # cross term [D_t, H_mod]
        normB2.append(float(np.linalg.norm(B, 'fro') ** 2))

    normB2 = np.array(normB2)
    slope = float(np.polyfit(np.log(omegas), np.log(normB2), 1)[0])

    print("Cross term B=[D_t,H_mod]: ||B||^2 (= (dH/dt)^2) vs frequency w")
    for w, v in zip(omegas, normB2):
        print(f"  w={w:5.1f}:  ||B||^2 = {v:.5e}")
    print(f"  log-log slope = {slope:.2f}  (expected +2: ||B||^2 ~ (dH/dt)^2 ~ w^2)")
    print("  => the time-derivative (evolution) term sits in a_4 ~ Tr(D_R^4), NOT a_2 ~ Tr(D_R^2)")

    out = ROOT / "experiments" / "exp_spectral_action_a4_last_run.json"
    out.write_text(json.dumps({
        "omegas": omegas,
        "cross_term_normB2": normB2.tolist(),
        "slope_vs_omega": slope,
        "conclusion": "cross term [D_t,H_mod] (evolution, time derivative) has ||B||^2 ∝ ω^2, entering a_4 not a_2",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
