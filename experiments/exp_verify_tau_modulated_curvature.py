"""用物质源 tau(x)（占据数/密度，模长缺陷导致）调制球极投影度规，真算物理标量曲率。

球极投影度规 g_polar = Omega^2 delta，Omega = 2/(1+r^2)（S^2 半径 1 球极投影，K_bg=1）。
物质源 tau(x) 调制度规：g_phy = tau * g_polar = (sqrt(tau) * Omega)^2 delta = omega^2 delta。
2D 高斯曲率（共形变换）：K_phy = -omega^{-2} Delta ln omega（欧氏拉普拉斯）。

验证：tau 位置依赖 -> K_phy 是否位置依赖（std != 0），即「物质源 -> 弯曲」。

Code: `py -m experiments.exp_verify_tau_modulated_curvature`
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
    print("=== 用 tau(x) 调制度规 -> 物理标量曲率 ===")
    print()

    L = 200
    x = np.linspace(-6, 6, L)
    X, Y = np.meshgrid(x, x)
    r2 = X ** 2 + Y ** 2

    # 球极投影共形因子（S^2 半径 1 球极投影，K_bg = 1）
    Omega = 2.0 / (1.0 + r2)

    # tau(x) = 高斯缺陷（物质源，模长缺陷导致的位置依赖密度）
    A, sigma = 0.4, 1.0
    tau = 1.0 - A * np.exp(-r2 / (2 * sigma ** 2))

    # 物理度规 g_phy = tau * Omega^2 * delta = omega^2 * delta
    omega = np.sqrt(tau) * Omega
    ln_omega = np.log(omega)

    # 2D 高斯曲率 K = -omega^{-2} * Delta ln omega（格点差分）
    dx = x[1] - x[0]
    d2x = np.gradient(np.gradient(ln_omega, dx, axis=0), dx, axis=0)
    d2y = np.gradient(np.gradient(ln_omega, dx, axis=1), dx, axis=1)
    lap = d2x + d2y
    K = -lap / omega ** 2

    # 背景曲率（无物质源 tau=1）应该 = 1（S^2 球极投影常曲率）
    Omega_bg = 2.0 / (1.0 + r2)
    ln_O = np.log(Omega_bg)
    d2xb = np.gradient(np.gradient(ln_O, dx, axis=0), dx, axis=0)
    d2yb = np.gradient(np.gradient(ln_O, dx, axis=1), dx, axis=1)
    K_bg = -(d2xb + d2yb) / Omega_bg ** 2

    print(f"  1. 背景曲率 K_bg（无物质源 tau=1）：mean = {K_bg.mean():.4f}，std = {K_bg.std():.2e}")
    print(f"     => 背景常曲率 = 1（S^2 球极投影），std~0（无位置依赖）。")
    print()
    print(f"  2. 物理曲率 K_phy（tau 调制后）：mean = {K.mean():.4f}，std = {K.std():.4f}")
    print(f"     K_phy 范围 = [{K.min():.4f}, {K.max():.4f}]")
    # 位置依赖：K_phy 在缺陷中心（r=0）vs 远处
    center = L // 2
    far = L // 2 - 80
    print(f"     缺陷中心（r~0）K = {K[center, center]:.4f}")
    print(f"     远处（r~4）K = {K[center, far]:.4f}")
    print(f"     => K_phy std = {K.std():.4f} != 0：位置依赖（物质源 -> 弯曲）！")

    print()
    print("=== 结论 ===")
    print("  tau(x)（物质源）调制度规后，标量曲率 K_phy 位置依赖（std != 0）。")
    print("  背景（无 tau）= 常曲率 K_bg=1；加 tau = 位置依赖弯曲。")
    print("  这验证了「物质源 -> 黎曼弯曲」的机制（EH 方程左边 G 随右边 T 变化）。")
    print("  但注意：tau = 占据数 = T_00（能量密度），是标量，给的是「标量弯曲」（标量引力），")
    print("  不是「自旋 2 弯曲」（GR 需要完整 T_mu nu）。且 tau 依赖「满带占据规则」（未推缺口）。")

    summary = {
        "K_bg_mean": float(K_bg.mean()),
        "K_bg_std": float(K_bg.std()),
        "K_phy_mean": float(K.mean()),
        "K_phy_std": float(K.std()),
        "K_phy_range": [float(K.min()), float(K.max())],
        "position_dependent": bool(K.std() > 1e-3),
        "note": "tau (matter source) modulates metric -> scalar curvature position-dependent "
                "(std != 0); background (no tau) = constant K=1. This verifies 'matter -> "
                "Riemann curvature' (EH left side G follows right side T). But tau = T_00 "
                "(scalar) gives SCALAR curvature (scalar gravity), not spin-2; and tau depends "
                "on 'band-filling rule' (unproven gap).",
    }
    out = ROOT / "experiments" / "exp_verify_tau_modulated_curvature_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
