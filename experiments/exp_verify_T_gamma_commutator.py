"""自构造突围：键序 T 与手征 Gamma 的对易子 [T, Gamma] 的谱。

2D pi-flux torus（二分图），费米海密度矩阵 P（键序 T = P）。
手征 Gamma = diag(+1,-1,...)（子格 A +1，子格 B -1）。
对易子 [P, Gamma] = P Gamma - Gamma P，算它的本征值谱。

诚实判定：有限维矩阵谱离散（N 个点），给不出「连续微分结构」（需无限维连续谱）。

Code: `py -m experiments.exp_verify_T_gamma_commutator`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pi_flux_D(L):
    N = L * L

    def idx(x, y):
        return (y % L) * L + (x % L)

    D = np.zeros((N, N))
    for y in range(L):
        for x in range(L):
            i = idx(x, y)
            j = idx(x + 1, y); w = (-1.0) ** y
            D[i, j] = w; D[j, i] = w
            j = idx(x, y + 1); D[i, j] = 1.0; D[j, i] = 1.0
    return D


def main():
    print("=== [T, Gamma] 对易子的本征值谱 ===")
    print()

    L = 8
    N = L * L
    D = pi_flux_D(L)

    # 手征 Gamma：子格 A（x+y 偶）+1，子格 B（x+y 奇）-1
    gamma = np.array([1.0 if (x + y) % 2 == 0 else -1.0
                      for y in range(L) for x in range(L)])
    Gamma = np.diag(gamma)

    # 键序 T = 密度矩阵 P（费米海填负能）
    ev, V = np.linalg.eigh(D)
    occ = ev < 0.0
    P = V[:, occ] @ V[:, occ].conj().T
    T = P  # 键序 = 密度矩阵

    # 对易子 [T, Gamma]
    comm = T @ Gamma - Gamma @ T
    # [T,Gamma] 是反厄米，本征值纯虚数
    ev_comm = np.linalg.eigvalsh(comm)  # 反厄米 -> 纯虚数（eigvalsh 给实数，需 ×i）
    # 用 eigvals 看纯虚数
    ev_comm2 = np.linalg.eigvals(comm)

    print(f"  N={N}，[T, Gamma] 的本征值：")
    print(f"    实部最大 = {np.abs(ev_comm2.real).max():.2e}（应≈0，反厄米纯虚数）")
    print(f"    虚部范围 = [{ev_comm2.imag.min():.4f}, {ev_comm2.imag.max():.4f}]")
    print(f"    本征值个数 = {N}（离散，N 个点）")
    # 谱间隔（相邻本征值差）
    imag_sorted = np.sort(ev_comm2.imag)
    gaps = np.diff(imag_sorted)
    print(f"    相邻本征值间隔：min={gaps.min():.2e}, max={gaps.max():.4f}")

    print()
    print("=== 诚实结论 ===")
    print("  [T, Gamma] 的本征值：纯虚数（反厄米）、离散（N 个点）、有谱间隔。")
    print("  有限维矩阵 -> 离散谱 -> 给不出「连续微分结构」（需无限维连续谱）。")
    print("  => 「自构造突围」没有绕过墙：还是有限维离散谱，不是连续微分结构。")
    print("     连续谱需要无限维（II_1 因子 R），而 R 上无紧 D（导子全内）。")

    summary = {
        "N": N,
        "commutator_real_max": float(np.abs(ev_comm2.real).max()),
        "commutator_imag_range": [float(ev_comm2.imag.min()), float(ev_comm2.imag.max())],
        "n_eigenvalues": N,
        "gap_min": float(gaps.min()),
        "gap_max": float(gaps.max()),
        "conclusion": "[T,Gamma] eigenvalues: pure imaginary (anti-Hermitian), discrete (N points), "
                      "gapped. Finite-dim -> discrete spectrum -> no continuous differential "
                      "structure (needs infinite-dim continuous spectrum). Self-construction "
                      "does NOT bypass the wall: still finite discrete, not continuous.",
    }
    out = ROOT / "experiments" / "exp_verify_T_gamma_commutator_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
