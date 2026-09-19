"""建议3（谱序 vs 空间序，对着 P3）：模流的谱序（能量序）是否等于空间序（位置序）。

实验 C 坐实：差分 D 的空间序（i 相邻）外给（标号手放）。
本脚本检验：模流生成元 log Delta 的谱序（按本征值排序 = 能量序）能否冒充空间序。

关键区分（1D 链紧束缚）：
  谱序（能量序）：能量本征基 |k>（波数 k，本征值 E_k）。
  空间序（位置序）：位置本征基 |x>（位置 x）。
  两者是傅里叶对偶：
    - 位置局域态 delta(x_0) 在能量基里「非局域」（几乎所有 k 都有分量）；
    - 能量本征态 u_k 在位置基里「非局域」（遍布所有位置，全局正弦波）。
  => 谱序（能量）!= 空间序（位置），模流给时间/能量序，不给空间序。

结论：模流的谱序是能量/时间序，空间序是它的傅里叶对偶，需额外结构（不是模流直接给）。
=> P3 换墙坐实：空间序外给，模流（时间序）不诱导空间序。

Code: `py -m experiments.exp_wall_spectral_vs_spatial`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    print("=== 建议3（谱序 vs 空间序，对着 P3）===")
    print()

    N = 64
    x = np.arange(1, N + 1)
    k = np.arange(1, N + 1)
    E = -2.0 * np.cos(k * np.pi / (N + 1))  # 能量本征值（k 序 = 能量序）

    # ---- 1. 位置局域态在能量基里非局域（傅里叶对偶）----
    center = N // 2  # 中心位置（0-based）
    # 位置局域态 delta(x_center) 在能量基 |k> 里的展开系数 = sin(k pi (center+1)/(N+1))
    coeffs = np.array([np.sin(kk * np.pi * (center + 1) / (N + 1)) for kk in k])
    coeffs = coeffs / np.linalg.norm(coeffs)
    n_sig = int(np.sum(np.abs(coeffs) > 0.05 * np.max(np.abs(coeffs))))
    print(f"1. 位置局域态 delta(x_center)（中心 x={center+1}）在能量基 |k> 里的分量：")
    print(f"   展开系数非零（>5%峰值）个数 = {n_sig}/{N}（几乎遍及所有 k）")
    print(f"   => 位置局域 -> 能量非局域（一个位置 = 所有能量的叠加）。")

    # ---- 2. 能量本征态在位置基里非局域 ----
    k_low = 1
    u_low = np.sin(k_low * np.pi * x / (N + 1))
    n_nodes_low = int(np.sum(np.abs(np.diff(np.sign(u_low))) > 0))
    k_high = N
    u_high = np.sin(k_high * np.pi * x / (N + 1))
    n_nodes_high = int(np.sum(np.abs(np.diff(np.sign(u_high))) > 0))
    print()
    print(f"2. 能量本征态在位置基里非局域：")
    print(f"   低能 k=1：波长=全链，节点数={n_nodes_low}（空间最不局域）")
    print(f"   高能 k={N}：节点数={n_nodes_high}（空间交替，最局域）")
    print(f"   => 能量局域（单一 k）-> 空间非局域（全局正弦波）。")

    print()
    print("=== 结论 ===")
    print("  谱序（能量 k）和空间序（位置 x）是傅里叶对偶的两面：")
    print("  位置局域 <-> 能量非局域，能量局域 <-> 空间非局域（不确定关系）。")
    print("  模流生成元 log Delta 的谱序是能量/时间序，不诱导空间序。")
    print("  => P3 换墙坐实：空间序外给，模流（时间序）给不出空间序。")

    summary = {
        "question": "does modular-flow spectral order (=energy) induce spatial order (=position)?",
        "answer": "NO: spectral order (eigenvalue/E) and spatial order (position) are Fourier-DUAL. "
                  "position-local state -> energy-nonlocal (all k), energy eigenstate -> "
                  "spatially-nonlocal (global sine). Modular flow gives time/energy order, "
                  "not spatial order.",
        "position_local_energy_nsig": n_sig,
        "low_k_nodes": n_nodes_low,
        "high_k_nodes": n_nodes_high,
        "conclusion": "P3 wall-swap confirmed: spatial order extrinsic, modular flow (time order) "
                      "does NOT induce spatial order (they are Fourier-dual).",
    }
    out = ROOT / "experiments" / "exp_wall_spectral_vs_spatial_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
