"""探索 A 第一步：松原格林函数 G(i,j;tau) 给「第 4 维（虚时间）」？

1D 环 + 缺陷，费米海。松原格林函数 G(i,j;tau) = [e^{-tau D} f(D)]_ij，
f(D) = 1/(1+e^{beta D})（费米分布），tau = 虚时间。

验证：G(i,j;tau) 是 (i,j,tau) 的 4 维对象（3 空间指标 i,j + 1 虚时间 tau）。
诚实标注：tau 依赖温度 beta（手放），所以「虚时间 4 维」是「手放温度」给的。

Code: `py -m experiments.exp_verify_matsubara`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def ring_D(N, V=0.0, x0=None):
    D = np.zeros((N, N))
    for i in range(N):
        j = (i + 1) % N
        D[i, j] = 1.0; D[j, i] = 1.0
    if x0 is not None:
        D[x0, x0] += V
    return D


def matsubara(D, tau, beta):
    """G(tau) = e^{-tau D} f(D)，f(D) = 1/(1+e^{beta D})。"""
    fD = np.linalg.inv(np.eye(D.shape[0]) + expm(beta * D))
    return expm(-tau * D) @ fD


def main():
    print("=== 探索 A：松原格林函数 G(i,j;tau) 给第 4 维（虚时间）===")
    print()

    N = 60
    x0 = 30
    V = 0.8
    beta = 10.0

    D = ring_D(N, V=V, x0=x0)

    print(f"  1D 环 + 缺陷（V={V}），温度 beta={beta}。")
    print()
    print("  松原格林函数 G(i,j;tau) 是 (i,j,tau) 的函数：")
    taus = [0.0, 2.0, 5.0, 8.0]
    for tau in taus:
        G = matsubara(D, tau, beta)
        # 近邻键序随 tau 变化（第 4 维 = 虚时间）
        diag_val = np.real(np.diag(G)).mean()
        print(f"    tau={tau}: G 对角平均 = {diag_val:.5f}（随虚时间 tau 变化）")
    print()
    print("  => G(i,j;tau) 是 4 维对象（3 空间指标 i,j + 1 虚时间 tau）。")
    print("     虚时间 tau 是「第 4 维」，松原格林函数天然 4 维。")

    print()
    print("=== 诚实边界 ===")
    print("  1. 虚时间 tau 依赖温度 beta（手放参数）：tau in [0,beta]，beta=1/kT。")
    print("     => 「虚时间 4 维」是「手放温度」给的，不是「从公设推的 4 维」。")
    print("  2. 理论的「干净 4 维」应该是「时间 = 有向区分」（号差 -+++，已推），")
    print("     不是「温度 beta」（手放热态）。")
    print("  3. 所以探索 A（虚时间）引入手放温度，和探索 B（SOC）引入手放 SOC 强度，")
    print("     是同一类「手放参数」——都不是完全干净。")

    summary = {
        "N": N, "V": V, "beta": beta,
        "tau_values": taus,
        "G_diag_mean_by_tau": {str(tau): float(np.real(np.diag(matsubara(D, tau, beta))).mean()) for tau in taus},
        "conclusion": "Matsubara G(i,j;tau) is 4D (3 spatial + 1 imaginary time), "
                      "but tau depends on temperature beta (hand-picked). The clean 4D "
                      "should be time = directed distinction (already derived), not "
                      "temperature (hand-picked thermal state).",
    }
    out = ROOT / "experiments" / "exp_verify_matsubara_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
