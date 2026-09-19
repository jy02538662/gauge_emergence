"""第一步（炸墙 ③）：局域高斯波包采样 -> Fisher 度规 G(X)，验证「参数 = 物理坐标」。

1D 环，局域高斯波包 |Phi(X)> = sum_i e^{-(i-X)^2/2sigma^2} |i>（归一化）。
采样态是纯态，Bures 距离 d_B^2 = 2 - 2|<Phi(X)|Phi(X+dX)>|。
Fisher 度规 G(X) = 对 X 的二阶导。

Code: `py -m experiments.exp_coordinate_sampling`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def gaussian(N, X, sigma):
    i = np.arange(N)
    psi = np.exp(-(i - X) ** 2 / (2 * sigma ** 2))
    return psi / np.linalg.norm(psi)


def main():
    print("=== 第一步：局域高斯波包采样 -> Fisher 度规 G(X) ===")
    print()

    N = 100
    sigma = 3.0

    def overlap(X1, X2):
        return abs(np.dot(gaussian(N, X1, sigma), gaussian(N, X2, sigma)))

    def bures2(X1, X2):
        return 2.0 - 2.0 * overlap(X1, X2)

    Xs = np.linspace(10, N - 10, 30)
    G = np.zeros(len(Xs))
    dX = 0.5
    for idx, X in enumerate(Xs):
        d2 = (bures2(X, X + dX) - 2 * bures2(X, X) + bures2(X, X - dX)) / dX ** 2
        G[idx] = 0.5 * d2

    print(f"  局域高斯波包（sigma={sigma}）采样：")
    print(f"  Fisher 度规 G(X) 随物理坐标 X：mean = {G.mean():.6f}, std = {G.std():.2e}")
    print(f"  G(X) 范围 = [{G.min():.6f}, {G.max():.6f}]")
    print(f"  => 自变量 = 物理坐标 X（参数 = 物理，炸墙 ③）。")

    # 对比：G 是否位置依赖（均匀环，高斯波包平移不变，G 应该近似常数）
    print()
    print("  诚实观察：均匀环 + 平移不变高斯波包 -> G(X) 近似常数（平移不变）。")
    print("  要 G(X) 位置依赖，需要「缺陷」打破平移不变（物质源）。")

    print()
    print("=== 诚实边界 ===")
    print("  1D 采样（纯空间 1 维），Fisher 度规 1x1，无 Weyl（1 维恒平）。")
    print("  要 Weyl/TT（炸墙 ②），需要 4 维（3 空间 + 1 时间）。")
    print("  第一步验证「参数 = 物理坐标」（炸墙 ③），但均匀环 G 平移不变，")
    print("  要位置依赖需「缺陷」（物质源，墙 ①）。")

    summary = {
        "N": N, "sigma": sigma,
        "G_mean": float(G.mean()),
        "G_std": float(G.std()),
        "G_range": [float(G.min()), float(G.max())],
        "parameter_is_physical": True,
        "note": "1D sampling: G(X) is function of physical X (parameter=physical, wall 3). "
                "But uniform ring -> G translation-invariant; need defect (matter) for "
                "position-dependence, and 4D for Weyl/TT.",
    }
    out = ROOT / "experiments" / "exp_coordinate_sampling_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
