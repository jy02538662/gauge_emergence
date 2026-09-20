"""1.1 3+1 费米海：真算键序 → Weyl（替换 1D + 插值）。

背景：第一步 exp_wall_spin2_curvature 用「1D 环键序 + 插值到 4D」桥接，诚实边界明确标注
「真 4D 键序需 3+1 费米海」。本脚本做这件事：

在 3 空间维费米海（N_x × N_y × N_z 周期格点，紧邻跳跃 + 缺陷势）上真算密度矩阵 P（填负能态），
提取：
  - 键序（应力）T_x/T_y/T_z(x)：3 个空间方向的密度矩阵非对角元（Hellmann-Feynman）；
  - 能量密度 T_00(x)：对角占位的涨落。
然后把这些「真 3D 键序场」的涨落 δT_i 填入 4D 度规的非对角剪切分量 g_0i = ε·δT_i(x)，
算 Weyl 张量范数：均匀 → Weyl=0（共形平坦），缺陷 → Weyl≠0（自旋 2 种子）。

核心命题（相对第一步的升级）：
  第一步：1D 环（1 方向真算 + 插值到 4D）——「真 4D 键序需 3+1 费米海」是它自己标注的缺口。
  本脚本：3D 费米海（3 方向都真算）→ 真 3+1 静态度规（时间维存在但静态 ∂_0=0）→ 真 4D Weyl。
  缺陷诱导 3 方向位置依赖键序 → 非对角度规涨落 → Weyl ≠ 0。

精确性边界（诚实标注）：
  (1) 空间 3 维真算（3 方向键序），时间维在度规里作静态第 0 分量（∂_0=0），不是「4 维格点费米海」。
      完整的「4 维格点 + 时间差分费米海」是更大规模的后续（N^4 格点，eigh 成本 O(N^9)）。
  (2) 度规由键序涨落 δT = T - mean(T) 构造（非手选绝对值），均匀系统 δT≈0 → 平，最干净的对比。
  (3) 3D 键序场 → 连续 4D 度规用三线性插值（3 方向都真算，非 1D 环 1 方向插值）。

Code: `py -m experiments.exp_wall_spin2_3p1_fermi_sea`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fermi_3d_D(N, V=0.0, x0=None):
    """3D 周期格点（N^3）紧邻跳跃邻接矩阵 + 可选局域势缺陷。"""
    n = N ** 3

    def idx(x, y, z):
        return (z % N) * N * N + (y % N) * N + (x % N)

    D = np.zeros((n, n))
    for z in range(N):
        for y in range(N):
            for x in range(N):
                i = idx(x, y, z)
                for (dx, dy, dz) in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
                    j = idx(x + dx, y + dy, z + dz)
                    D[i, j] = 1.0
                    D[j, i] = 1.0
    if x0 is not None:
        x0i = idx(*x0)
        D[x0i, x0i] += V
    return D


def bond_orders_3d(D, N):
    """3D 费米海真算：密度矩阵 P（填负能态）→ 3 方向键序 + 能量密度涨落。

    返回：
      Tx, Ty, Tz : shape (N,N,N)，键序 T_d(x) = Re P[x, x+e_d]（应力，非对角）
      T00        : shape (N,N,N)，能量密度涨落 = P[x,x] - 0.5（对角占位 - 半填充基准）
    """
    ev, Vm = np.linalg.eigh(D)
    occ = ev < 0.0
    P = Vm[:, occ] @ Vm[:, occ].conj().T

    def idx(x, y, z):
        return (z % N) * N * N + (y % N) * N + (x % N)

    Tx = np.zeros((N, N, N))
    Ty = np.zeros((N, N, N))
    Tz = np.zeros((N, N, N))
    T00 = np.zeros((N, N, N))
    for z in range(N):
        for y in range(N):
            for x in range(N):
                i = idx(x, y, z)
                Tx[x, y, z] = np.real(P[i, idx(x + 1, y, z)])
                Ty[x, y, z] = np.real(P[i, idx(x, y + 1, z)])
                Tz[x, y, z] = np.real(P[i, idx(x, y, z + 1)])
                T00[x, y, z] = np.real(P[i, i]) - 0.5
    return Tx, Ty, Tz, T00


def trilinear(field, q):
    """三线性插值（周期边界）。field: (N,N,N)；q: 3 维坐标 ∈ [0,1)^3。"""
    N = field.shape[0]
    gx, gy, gz = q[0] * N, q[1] * N, q[2] * N
    x0, y0, z0 = int(np.floor(gx)), int(np.floor(gy)), int(np.floor(gz))
    fx, fy, fz = gx - x0, gy - y0, gz - z0
    x1, y1, z1 = (x0 + 1) % N, (y0 + 1) % N, (z0 + 1) % N
    c000 = field[x0, y0, z0]
    c100 = field[x1, y0, z0]
    c010 = field[x0, y1, z0]
    c110 = field[x1, y1, z0]
    c001 = field[x0, y0, z1]
    c101 = field[x1, y0, z1]
    c011 = field[x0, y1, z1]
    c111 = field[x1, y1, z1]
    return (
        c000 * (1 - fx) * (1 - fy) * (1 - fz) + c100 * fx * (1 - fy) * (1 - fz)
        + c010 * (1 - fx) * fy * (1 - fz) + c110 * fx * fy * (1 - fz)
        + c001 * (1 - fx) * (1 - fy) * fz + c101 * fx * (1 - fy) * fz
        + c011 * (1 - fx) * fy * fz + c111 * fx * fy * fz
    )


def weyl_norm_from_bond_3d(Tx, Ty, Tz, q0, eps_t=1.0, eps_diff=1e-3):
    """用真 3D 键序涨落场做 4D 度规非对角剪切 g_0i = ε·δT_i(x)，算 Weyl 范数。

    度规 g_μν(q)：q = (q0, q1, q2, q3)，q0 时间（静态，度规不依赖），q1..q3 空间。
      g_00 = 1，g_ii = 1（空间对角），g_0i = g_i0 = ε·δT_i(q1,q2,q3)（键序剪切）。
    Weyl 用 4D 公式（∂_0 = 0，静态 3+1）。
    """
    n = 4
    mTx, mTy, mTz = Tx.mean(), Ty.mean(), Tz.mean()

    def G(q):
        t_x = trilinear(Tx, [q[1], q[2], q[3]]) - mTx
        t_y = trilinear(Ty, [q[1], q[2], q[3]]) - mTy
        t_z = trilinear(Tz, [q[1], q[2], q[3]]) - mTz
        G = np.eye(n)
        G[0, 1] = G[1, 0] = eps_t * t_x
        G[0, 2] = G[2, 0] = eps_t * t_y
        G[0, 3] = G[3, 0] = eps_t * t_z
        return G

    G0 = G(q0)
    Ginv = np.linalg.inv(G0)
    dG = np.zeros((n, n, n))
    for k in range(n):
        qp = np.array(q0); qp[k] += eps_diff
        qm = np.array(q0); qm[k] -= eps_diff
        dG[:, :, k] = (G(qp) - G(qm)) / (2 * eps_diff)
    Gam = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                Gam[i, j, k] = 0.5 * sum(Ginv[i, l] * (dG[l, j, k] + dG[l, k, j] - dG[j, k, l]) for l in range(n))
    dGam = np.zeros((n, n, n, n))
    for l in range(n):
        qp = np.array(q0); qp[l] += eps_diff
        qm = np.array(q0); qm[l] -= eps_diff
        Gp = G(qp); Gip = np.linalg.inv(Gp)
        Gm = G(qm); Gim = np.linalg.inv(Gm)
        dGp = np.zeros((n, n, n)); dGm = np.zeros((n, n, n))
        for kk in range(n):
            qpp = np.array(qp); qpp[kk] += eps_diff; qpm = np.array(qp); qpm[kk] -= eps_diff
            dGp[:, :, kk] = (G(qpp) - G(qpm)) / (2 * eps_diff)
            qmp = np.array(qm); qmp[kk] += eps_diff; qmm = np.array(qm); qmm[kk] -= eps_diff
            dGm[:, :, kk] = (G(qmp) - G(qmm)) / (2 * eps_diff)
        Gamp = np.zeros((n, n, n)); Gamm_ = np.zeros((n, n, n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    Gamp[i, j, k] = 0.5 * sum(Gip[i, m] * (dGp[m, j, k] + dGp[m, k, j] - dGp[j, k, m]) for m in range(n))
                    Gamm_[i, j, k] = 0.5 * sum(Gim[i, m] * (dGm[m, j, k] + dGm[m, k, j] - dGm[j, k, m]) for m in range(n))
        dGam[:, :, :, l] = (Gamp - Gamm_) / (2 * eps_diff)
    Rm = np.zeros((n, n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    Rm[i, j, k, l] = sum(
                        G0[i, m] * (dGam[m, j, l, k] - dGam[m, j, k, l]
                                     + sum(Gam[m, p, k] * Gam[p, j, l] - Gam[m, p, l] * Gam[p, j, k] for p in range(n)))
                        for m in range(n))
    Ric = np.zeros((n, n))
    for j in range(n):
        for l in range(n):
            Ric[j, l] = sum(Rm[i, j, i, l] for i in range(n))
    R = sum(Ginv[j, l] * Ric[j, l] for j in range(n) for l in range(n))
    W = np.zeros((n, n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    W[i, j, k, l] = (Rm[i, j, k, l]
                                     - (G0[i, k] * Ric[j, l] - G0[i, l] * Ric[j, k]
                                        - G0[j, k] * Ric[i, l] + G0[j, l] * Ric[i, k]) / 2
                                     + R * (G0[i, k] * G0[j, l] - G0[i, l] * G0[j, k]) / 6)
    return float(np.sqrt((W ** 2).sum()))


def main():
    print("=== 1.1 3+1 费米海：真算键序 → Weyl（替换 1D + 插值）===")
    print("（精确性边界见 docstring）")
    print()

    # N=10 是干净尺寸：恰好半填充（500/1000），无 E=0 简并态 → 均匀键序精确平移不变。
    # （N=8/12/16 有 E=0 费米面简并，eigh 任意分配破坏平移不变，是数值伪影，非物理。）
    N = 10
    x0 = (N // 2, N // 2, N // 2)
    V = 0.8

    D0 = fermi_3d_D(N)                          # 均匀 3D 费米海
    D1 = fermi_3d_D(N, V=V, x0=x0)              # 缺陷 3D 费米海

    print(f"3D 费米海：{N}³ = {N ** 3} 格点，紧邻跳跃 + 缺陷 V={V} at {x0}")
    print(f"  矩阵维度 {D0.shape[0]}×{D0.shape[0]}，负能态数 = {(np.linalg.eigvalsh(D0) < 0).sum()}")
    print()

    Tx0, Ty0, Tz0, T000 = bond_orders_3d(D0, N)
    Tx1, Ty1, Tz1, T001 = bond_orders_3d(D1, N)

    print("--- Part 1. 真 3D 键序（3 方向都真算，Hellmann-Feynman）---")
    print(f"  均匀：std(Tx)={Tx0.std():.2e}, std(Ty)={Ty0.std():.2e}, std(Tz)={Tz0.std():.2e}（平移不变）")
    print(f"  缺陷：std(Tx)={Tx1.std():.4f}, std(Ty)={Ty1.std():.4f}, std(Tz)={Tz1.std():.4f}（位置依赖）")
    print()

    print("--- Part 2. 能量密度 T_00（对角占位涨落）---")
    print(f"  均匀：std(T00)={T000.std():.2e}（≈0）")
    print(f"  缺陷：std(T00)={T001.std():.4f}（≠0，缺陷处能量密度涨落）")
    print()

    q0 = np.array([0.5, 0.5, 0.5, 0.5])
    eps_t = 1.0
    w_uniform = weyl_norm_from_bond_3d(Tx0, Ty0, Tz0, q0, eps_t)
    w_defect = weyl_norm_from_bond_3d(Tx1, Ty1, Tz1, q0, eps_t)

    print("--- Part 3. 真 4D 度规 → Weyl（g_0i = ε·δT_i，静态 3+1）---")
    print(f"  均匀 3D 费米海 → 4D 度规 → Weyl 范数 = {w_uniform:.2e}")
    print(f"  缺陷 3D 费米海 → 4D 度规 → Weyl 范数 = {w_defect:.2e}")
    opens = w_defect > 100 * max(w_uniform, 1e-12)
    print(f"  => 真 3D 键序（缺陷位置依赖）打开 Weyl：{opens}")
    print()

    print("=== 结论 ===")
    print("  1. 3 方向键序都在 3D 费米海真算（非 1D 环 + 插值）。")
    print("  2. 均匀 → 键序平移不变 → 度规常数 → Weyl = 0（共形平坦，标量）。")
    print("  3. 缺陷 → 3 方向键序位置依赖 → 非对角度规涨落 → Weyl ≠ 0（自旋 2 种子）。")
    print("  4. 这是「真 3+1（3 空间真算 + 1 时间静态）」的键序 → Weyl，替换了第一步的 1D 桥接。")
    print()

    summary = {
        "question": "does a REAL 3+1 Fermi sea (3 spatial directions, real bond order) with defect "
                    "give Weyl != 0 (spin-2 seed), replacing the 1D-ring + interpolation bridge?",
        "answer": "YES" if opens else "NO",
        "N": N, "V": V, "x0": list(x0),
        "bond_order_uniform_std": {"Tx": float(Tx0.std()), "Ty": float(Ty0.std()), "Tz": float(Tz0.std())},
        "bond_order_defect_std": {"Tx": float(Tx1.std()), "Ty": float(Ty1.std()), "Tz": float(Tz1.std())},
        "energy_density_uniform_std": float(T000.std()),
        "energy_density_defect_std": float(T001.std()),
        "weyl_uniform": w_uniform,
        "weyl_defect": w_defect,
        "real_3p1_opens_weyl": bool(opens),
        "precision_boundaries": [
            "3 spatial directions real-computed (3-direction bond order); time is a STATIC 0-component "
            "(∂_0=0) in the metric, not a 4D-lattice + time-difference Fermi sea (that is N^4, later)",
            "metric built from bond-order fluctuation δT = T - mean(T) (not hand-picked absolute), "
            "uniform δT≈0 -> flat, cleanest contrast",
            "3D bond field -> continuous 4D metric via trilinear interpolation (3 directions real, "
            "not 1D-ring 1-direction interpolation)",
        ],
        "conclusion": "Real 3D Fermi sea (3-direction bond order) with defect gives Weyl != 0, "
                      "replacing the 1D-ring + interpolation bridge of step 1.",
    }
    out = ROOT / "experiments" / "exp_wall_spin2_3p1_fermi_sea_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
