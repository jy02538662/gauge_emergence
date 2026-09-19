"""检验：傅里叶对偶保维（模流频率一维），3 维来自 D 的谱维数（升维）。

核心命题：
  傅里叶对偶是「保维」的：时间 t（一维）↔ 频率 ω（一维）。
  所以「模流（一维时间）→ 空间」的傅里叶对偶给「一维空间」，不给 3 维。
  「3 维」来自 D 的谱维数 d_s=3（本征值密度幂律），是「升维」，不是「对偶」。

坐实两点（numpy 数值）：
  1. 模流频率 θ_ij = log(λ_i/λ_j) 是低秩（一维参数化 s = log λ）。
  2. D 的谱维数 d_s 从本征值密度幂律读出（1D 环 -> d_s=1，3D 环 -> d_s=3）。

结论：时间（模流）一维、空间（D 谱）3 维，「升维」= 谱维数（付费桥 2），不是傅里叶对偶。

Code: `py -m experiments.exp_wall_fourier_preserves_dimension`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np.random.seed(0)


def spectral_dimension(L, n_low=30):
    """累积分布 N(λ) ~ λ^{d_s/2}，取最低 n_low 个非零本征值拟合（低能渐近）。"""
    evals = np.linalg.eigvalsh(L)
    evals = np.sort(evals[evals > 1e-6])  # 去零模
    evals = evals[:n_low]
    logx = np.log(evals)
    logy = np.log(np.arange(1, len(evals) + 1))
    slope = np.polyfit(logx, logy, 1)[0]
    return 2.0 * slope


def laplacian_1d(N):
    """周期环 Laplacian（1D），本征值 2 - 2cos(2πk/N)。"""
    L = np.zeros((N, N))
    for i in range(N):
        L[i, i] = 2.0
        L[i, (i + 1) % N] = -1.0
        L[i, (i - 1) % N] = -1.0
    return L


def laplacian_3d(n):
    """周期环 Laplacian（3D），N = n^3。"""
    L1 = laplacian_1d(n)
    I = np.eye(n)
    L = (np.kron(np.kron(L1, I), I) + np.kron(np.kron(I, L1), I)
         + np.kron(np.kron(I, I), L1))
    return L


def main():
    print("=== 傅里叶对偶保维 + 3 维来自谱维数（数值坐实）===")
    print()

    # ---- 1. 模流频率低秩（一维参数化）----
    print("1. 模流频率 θ_ij = log(λ_i/λ_j) 低秩（一维参数化 s = log λ）")
    N = 32
    lam = 1.0 / np.arange(1, N + 1)          # λ_i = 1/i（尺度不变观察者态）
    theta = np.log(lam[:, None] / lam[None, :])  # θ_ij = log(λ_i/λ_j)
    sv = np.linalg.svd(theta, compute_uv=False)
    # θ_ij = s_j - s_i（s = log λ），反对称，非零奇异值 = 1 对（秩 2）
    n_nonzero = int(np.sum(sv > 1e-8 * sv[0]))
    print(f"   N={N}：θ 矩阵的奇异值非零个数 = {n_nonzero}（满秩应为 {N}）")
    print(f"   奇异值前 5 = {np.round(sv[:5], 3)}")
    print(f"   -> θ 由一维序列 s=log λ 决定（θ_ij=s_j-s_i），低秩 = 一维参数化 [{'OK' if n_nonzero <= 2 else 'FAIL'}]")

    # ---- 2. D 的谱维数（1D vs 3D）----
    print()
    print("2. D 的谱维数 d_s（本征值密度幂律）：1D 环 -> 1，3D 环 -> 3")
    for name, L, expect in [("1D 环", laplacian_1d(200), 1.0),
                            ("3D 环 (8^3)", laplacian_3d(8), 3.0)]:
        ds = spectral_dimension(L)
        print(f"   {name}：d_s = {ds:.2f}（期望 {expect:.0f}）")
        print(f"       -> {'[OK]' if abs(ds - expect) < 0.7 else '[FAIL]'}")

    # ---- 结论 ----
    print()
    print("=== 结论 ===")
    print("  模流（时间）频率 θ_ij = s_j - s_i 由一维 s=log λ 决定（低秩）。")
    print("  傅里叶对偶保维：时间（一维）<-> 频率（一维），给不出 3 维。")
    print("  3 维来自 D 的谱维数 d_s=3（本征值密度幂律），是「升维」，不是「对偶」。")
    print("  => 「一维模流 → 3 维空间」的「3」= 谱维数（付费桥 2），不是傅里叶对偶。")

    summary = {
        "question": "does Fourier duality preserve dimension? does '3D' come from spectral dimension?",
        "modular_freq_rank": int(n_nonzero),
        "modular_freq_expected_rank": 2,
        "spectral_dim_1d": float(spectral_dimension(laplacian_1d(200))),
        "spectral_dim_3d": float(spectral_dimension(laplacian_3d(8))),
        "conclusion": "modular (time) frequencies theta_ij = s_j - s_i determined by 1D s=log lambda "
                      "(low rank). Fourier duality preserves dimension (1D time <-> 1D frequency). "
                      "3D comes from spectral dimension d_s=3 (eigenvalue-density power law) = 'dimension "
                      "lifting' (paid-bridge-2), NOT from Fourier duality.",
    }
    out = ROOT / "experiments" / "exp_wall_fourier_preserves_dimension_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
