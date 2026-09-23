"""4D: does the evolution (time derivative) enter a_2 (scalar curvature)?

对照 2D 结论（演化项在 a_4，因 Tr σ_y = 0 使交叉项一阶为零），4D 里演化项进 a_2——
因为 a_2 = (1/6)∫R，而标量曲率 R 是度规的二阶导（含 ∂_t²）。用 FLRW 度规符号验证。

命题（「整体动力」4D 有限目标）：4D 载体 D_{3+1} 的谱作用量，热核 a_2 = (1/6)∫R，
R 含时间导数（动态度规），所以演化项进 a_2 —— 即 4D 里「约束 + 演化」同源在 a_2，
与 2D 里演化项在 a_4 不同。这是「4D GR 演化方程」的机制来源。
"""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    t, x, y, z = sp.symbols("t x y z", real=True)
    a = sp.Function("a")(t)  # scale factor a(t)

    # FLRW metric g = diag(-1, a(t)^2, a(t)^2, a(t)^2), coords (t,x,y,z)
    coords = [t, x, y, z]
    g = sp.diag(-1, a**2, a**2, a**2)
    g_inv = sp.diag(-1, a ** (-2), a ** (-2), a ** (-2))

    n = 4
    # Christoffel Γ^ρ_{μν}
    Gamma = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for rho in range(n):
        for mu in range(n):
            for nu in range(n):
                s = 0
                for sig in range(n):
                    s += sp.Rational(1, 2) * g_inv[rho, sig] * (
                        sp.diff(g[sig, nu], coords[mu])
                        + sp.diff(g[sig, mu], coords[nu])
                        - sp.diff(g[mu, nu], coords[sig])
                    )
                Gamma[rho][mu][nu] = sp.simplify(s)

    # Ricci R_{μν} = ∂_ρ Γ^ρ_{μν} - ∂_ν Γ^ρ_{μρ} + Γ^ρ_{ρσ} Γ^σ_{μν} - Γ^ρ_{σν} Γ^σ_{μρ}
    R_munu = [[0 for _ in range(n)] for _ in range(n)]
    for mu in range(n):
        for nu in range(n):
            s = 0
            for rho in range(n):
                s += sp.diff(Gamma[rho][mu][nu], coords[rho]) - sp.diff(Gamma[rho][mu][rho], coords[nu])
                for sig in range(n):
                    s += Gamma[rho][rho][sig] * Gamma[sig][mu][nu] - Gamma[rho][sig][nu] * Gamma[sig][mu][rho]
            R_munu[mu][nu] = sp.simplify(s)

    # scalar curvature R = g^{μν} R_{μν}
    R = sp.simplify(sum(g_inv[mu, nu] * R_munu[mu][nu] for mu in range(n) for nu in range(n)))

    # express in terms of a, adot, addot
    adot = sp.Function("adot")(t)
    addot = sp.Function("addot")(t)
    R_sub = sp.simplify(R.subs({sp.diff(a, t): adot, sp.diff(a, t, 2): addot}))

    # expected FLRW: R = 6 (ä/a + (ȧ/a)^2)
    R_expected = 6 * (addot / a + (adot / a) ** 2)
    match = bool(sp.simplify(R_sub - R_expected) == 0)

    print("4D FLRW scalar curvature R (does it contain time derivatives?)")
    print(f"  R = {sp.simplify(R)}")
    print(f"  R contains a'(t) and a''(t) time derivatives: {R_sub.has(adot) and R_sub.has(addot)}")
    print(f"  matches R = 6(a''/a + (a'/a)^2) = {match}")
    print("  => a_2 = (1/6)∫R contains time derivatives: evolution enters a_2 in 4D (unlike 2D where it's in a_4)")

    out = ROOT / "experiments" / "exp_4d_a2_last_run.json"
    out.write_text(json.dumps({
        "R": str(sp.simplify(R)),
        "R_has_time_derivatives": bool(R_sub.has(adot) and R_sub.has(addot)),
        "matches_FLRW_formula": match,
        "conclusion": "4D scalar curvature R contains ∂_t² (time derivative), so a_2 = (1/6)∫R carries evolution — unlike 2D (evolution in a_4)",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
