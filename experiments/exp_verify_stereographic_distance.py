"""球极投影下 S^3 测地距离 -> R^3 距离函数是否位置依赖。

符号 + 数值验证：球极投影（非平移不变）下，S^3 测地距离 d = arccos(x.y)
投影到 R^3 后，是否仅依赖欧氏距离 |y-y'|，还是也依赖位置 r=|y|？

Code: `py -m experiments.exp_verify_stereographic_distance`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    print("=== 球极投影下 S^3 测地距离是否位置依赖 ===")
    print()

    # ---- 符号：cos(d) 的表达式，验证它依赖 r（位置）----
    r, s = sp.symbols('r s', real=True, nonnegative=True)
    # 球极投影坐标下，S^3 内积 x.y = [(r^2-1)(s^2-1) + 4 y.y']/[(r^2+1)(s^2+1)]
    # 取 y, y' 在同一直线（y.y' = r*s，因为共线同向），欧氏距离 = |r-s|
    cos_d = ((r ** 2 - 1) * (s ** 2 - 1) + 4 * r * s) / ((r ** 2 + 1) * (s ** 2 + 1))
    cos_d = sp.simplify(cos_d)
    print("1. 符号：S^3 测地距离的 cos（球极投影坐标，共线点）：")
    print(f"   cos(d) = {cos_d}")
    print(f"   化简后 cos(d) = {sp.factor(cos_d)}")
    print("   => cos(d) 依赖 r 和 s（位置），不仅依赖 |r-s|（欧氏距离）。")

    # ---- 数值：相同欧氏距离，不同位置，测地距离是否不同 ----
    print()
    print("2. 数值：相同欧氏距离 a=0.5，不同位置 R，测地距离：")

    def geo_dist(y, yp):
        r2 = np.dot(y, y); s2 = np.dot(yp, yp); yd = np.dot(y, yp)
        cd = ((r2 - 1) * (s2 - 1) + 4 * yd) / ((r2 + 1) * (s2 + 1))
        return np.arccos(np.clip(cd, -1, 1))

    a = 0.5
    rows = []
    for R in (0.0, 0.5, 1.0, 2.0, 5.0):
        y = np.array([R, 0.0, 0.0])
        yp = np.array([R + a, 0.0, 0.0])
        d = geo_dist(y, yp)
        rows.append((R, float(d)))
        print(f"   R={R:>3}: 欧氏距离={a}, S^3 测地距离={d:.6f}")

    # 位置依赖：测地距离随 R 变化（相同欧氏距离）
    ds = [r[1] for r in rows]
    varies = max(ds) - min(ds) > 1e-6
    print(f"   => 相同欧氏距离，测地距离随位置 R 变化：{'是（位置依赖！）' if varies else '否（平移不变）'}")
    print(f"      范围 [{min(ds):.6f}, {max(ds):.6f}]")

    print()
    print("=== 结论 ===")
    print("  球极投影下，S^3 测地距离投影到 R^3 后，是「位置依赖」的：")
    print("  相同欧氏距离 |y-y'|，测地距离随位置 r=|y| 变化。")
    print("  这证明球极投影是「非平移不变」的（测地距离位置依赖）。")
    print("  但注意：这是「测地距离的位置依赖」，不等于「标量曲率的位置依赖」。")
    print("  （标量曲率 R=6 常数，是常曲率；测地距离位置依赖是共形因子的位置依赖。）")

    summary = {
        "cos_d_symbolic": str(sp.factor(cos_d)),
        "geodesic_distance_rows": [{"R": r[0], "d": r[1]} for r in rows],
        "position_dependent": bool(varies),
        "note": "geodesic distance IS position-dependent under stereographic projection "
                "(same Euclidean distance, different position -> different geodesic distance), "
                "BUT this is conformal-factor position-dependence, NOT scalar-curvature "
                "position-dependence (R=6 constant).",
    }
    out = ROOT / "experiments" / "exp_verify_stereographic_distance_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
