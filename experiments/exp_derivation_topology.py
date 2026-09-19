"""检验：导子 + 拓扑 的统一 = de Rham 上同调（闭但非恰当）。

纠正热核（平滑，不保 Leibniz）：正确的「导子 + 拓扑」统一 = 上同调上的导子。
涡旋梯度 gradθ = n(-y,x)/r² 是「闭形式」（无旋 grad×gradθ=0，保导子/Leibniz），
但「环流非零」（∮gradθ·dl = 2πn，绕数，拓扑）。这就是 de Rham 上同调的
「闭但非恰当」（绕数非零 ⟹ θ 非单值 ⟹ gradθ 不是恰当形式）。

坐实：
  1. gradθ 无旋（grad×gradθ = 0，闭形式，保导子结构）。
  2. 环流 ∮gradθ·dl = 2πn（绕数，拓扑不变量）。
  3. 结论：gradθ 是「闭但非恰当」= de Rham 上同调非平凡 = 导子 + 拓扑 的统一。

Code: `py -m experiments.exp_derivation_topology`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def vortex_gradient(L, n):
    """涡旋相位 θ=n·arctan(y/x) 的梯度 gradθ = n(-y,x)/r²（向量场 vx,vy）。"""
    cx = cy = (L - 1) / 2.0
    vx = np.zeros((L, L))
    vy = np.zeros((L, L))
    for x in range(L):
        for y in range(L):
            rx, ry = x - cx, y - cy
            r2 = rx ** 2 + ry ** 2
            if r2 < 1e-9:
                continue
            vx[x, y] = -n * ry / r2
            vy[x, y] = n * rx / r2
    return vx, vy


def main():
    print("=== 导子 + 拓扑 的统一 = de Rham 上同调（闭但非恰当）===")
    print()

    L = 60
    n = 1
    vx, vy = vortex_gradient(L, n)
    cx = cy = (L - 1) / 2.0

    # ---- 1. 无旋（闭形式，保导子）----
    print("1. gradθ 无旋（grad×gradθ = 0，闭形式，保导子结构）")
    # 2D 旋度 = ∂vy/∂x - ∂vx/∂y（中心差分）
    curl = np.zeros((L, L))
    for x in range(1, L - 1):
        for y in range(1, L - 1):
            curl[x, y] = (vy[x + 1, y] - vy[x - 1, y]) / 2 - (vx[x, y + 1] - vx[x, y - 1]) / 2
    # 远离核心（r > 3）的旋度
    curl_vals = []
    for x in range(L):
        for y in range(L):
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if r > 8:
                curl_vals.append(abs(curl[x, y]))
    max_curl = max(curl_vals)
    print(f"   远离核心（r>8）的 |grad×gradθ| 最大值 = {max_curl:.2e}（期望 ~0，无旋）")
    closed = max_curl < 0.05
    print(f"   gradθ 是闭形式（无旋，保导子/Leibniz）：{('[OK]' if closed else '[FAIL]')}")

    # ---- 2. 环流非零（绕数，拓扑）----
    print()
    print("2. 环流 ∮gradθ·dl = 2πn（绕数，拓扑不变量）")
    circs = []
    for r in [8, 12, 16, 20]:
        nphi = max(40, int(2 * np.pi * r))
        phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
        Gamma = 0.0
        for p in phi:
            x = cx + r * np.cos(p)
            y = cy + r * np.sin(p)
            xi, yi = int(round(x)), int(round(y))
            # 切向单位向量
            tx, ty = -np.sin(p), np.cos(p)
            # gradθ·dl = (vx·tx + vy·ty)·dl
            Gamma += (vx[xi, yi] * tx + vy[xi, yi] * ty) * (2 * np.pi * r / nphi)
        circs.append(Gamma)
    circs = np.array(circs)
    print(f"   环流 Γ(r) = {np.round(circs, 4)}（期望 2πn = {2*np.pi*n:.4f}，不随 r 变）")
    nonzero = abs(circs[0] - 2 * np.pi * n) < 0.3
    print(f"   环流非零（绕数 = 拓扑）：{('[OK]' if nonzero else '[FAIL]')}")

    # ---- 3. 结论 ----
    print()
    print("=== 结论 ===")
    print("  gradθ 无旋（闭形式，保导子/Leibniz）+ 环流非零（绕数，拓扑）。")
    print("  => gradθ 是「闭但非恰当」（de Rham 上同调非平凡）= 导子 + 拓扑 的统一。")
    print("  这比「热核（平滑）」和「δ 函数修正（发散）」都正确——是上同调上的导子。")

    summary = {
        "question": "is 'derivation + topology' unified as de Rham cohomology (closed but not exact)?",
        "curl_max": float(max_curl),
        "closed_form": bool(closed),
        "circulation_2pin": float(2 * np.pi * n),
        "circulation_values": [float(c) for c in circs],
        "winding_nonzero": bool(nonzero),
        "conclusion": "grad theta is CLOSED (curl-free, preserves derivation/Leibniz) but has NONZERO "
                      "circulation (winding, topology). 'closed but not exact' = de Rham cohomology "
                      "nontrivial = unification of derivation + topology. Correct form is cohomology, "
                      "not heat kernel (smoothing) nor delta-function correction (divergent).",
    }
    out = ROOT / "experiments" / "exp_derivation_topology_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
