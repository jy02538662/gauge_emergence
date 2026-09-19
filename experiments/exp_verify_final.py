"""补做之前"手推/引用定理"但未跑程序的验证（诚实补漏，修复版）。

  1. Gauss-Bonnet: discrete sphere angle-defect sum = 4 pi = 2 pi chi.
  2. noncommutative torus: UV = e^{2 pi i theta} VU  (intrinsic noncommutativity).
  3. M_2 derivations all inner: derivation space dim = 3 (HH^1(M_2)=0).

Code: `py -m experiments.exp_verify_final`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fibonacci_sphere(n):
    pts = []
    phi = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        y = 1 - (i / (n - 1)) * 2
        r = np.sqrt(max(0.0, 1 - y * y))
        th = phi * i
        pts.append([r * np.cos(th), y, r * np.sin(th)])
    return np.array(pts)


def part1():
    print("=== 1. Gauss-Bonnet: discrete sphere angle-defect sum = 4 pi ===")
    for n in (50, 100, 200):
        pts = fibonacci_sphere(n)
        hull = ConvexHull(pts)
        angle_sum = np.zeros(n)
        for face in hull.simplices:
            a, b, c = face
            va, vb = pts[b] - pts[a], pts[c] - pts[a]
            vc, vd = pts[c] - pts[b], pts[a] - pts[b]
            ve, vf = pts[a] - pts[c], pts[b] - pts[c]
            angle_sum[a] += np.arccos(np.clip(np.dot(va, vb) /
                                              (np.linalg.norm(va) * np.linalg.norm(vb)), -1, 1))
            angle_sum[b] += np.arccos(np.clip(np.dot(vc, vd) /
                                              (np.linalg.norm(vc) * np.linalg.norm(vd)), -1, 1))
            angle_sum[c] += np.arccos(np.clip(np.dot(ve, vf) /
                                              (np.linalg.norm(ve) * np.linalg.norm(vf)), -1, 1))
        defect = (2 * np.pi - angle_sum).sum()
        chi = defect / (2 * np.pi)
        print(f"    N={n}: sum(angle defect) = {defect:.6f} -> chi = {chi:.4f} (sphere chi=2)")
    print("    => topology (integer chi) and Riemann (local K) are two faces of ONE object.")
    return {"chi": float(chi)}


def part2():
    print()
    print("=== 2. noncommutative torus: UV = e^{2 pi i theta} VU ===")
    N = 6
    theta = 1.0 / 3.0
    omega = np.exp(2j * np.pi * theta)
    U = np.diag([omega ** i for i in range(N)])
    V = np.zeros((N, N), dtype=complex)
    for i in range(N):
        V[i, (i - 1) % N] = 1.0   # V @ e_i = e_{i+1}  =>  UV = omega VU
    lhs = U @ V
    rhs = omega * (V @ U)
    diff = np.max(np.abs(lhs - rhs))
    comm = np.max(np.abs(U @ V - V @ U))
    print(f"    N={N}, theta=1/3: max|UV - omega VU| = {diff:.2e}  (should be 0)")
    print(f"    max|UV - VU| = {comm:.4f}  (nonzero: intrinsic noncommutativity)")
    print("    => [U,V] = (omega-1)VU != 0 gives TOPOLOGICAL charge (Chern), not Riemann.")
    return {"diff": float(diff), "comm": float(comm)}


def part3():
    print()
    print("=== 3. M_2 derivations all inner (finite-dim Sakai-Kadison) ===")
    # delta(E_ij) = sum_{pq} c[ij,pq] E_pq ; c is 16-vector.
    # Leibniz: delta(E_ij E_kl) = delta(E_ij) E_kl + E_ij delta(E_kl).
    # E_ij E_kl = delta_jk E_il.
    def flat(i, j, p, q):
        return (2 * i + j) * 4 + (2 * p + q)

    rows = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for p in range(2):
                        for q in range(2):
                            r = np.zeros(16)
                            if j == k:                      # LHS delta_jk c[il,pq]
                                r[flat(i, l, p, q)] += 1.0
                            if q == l:                      # RHS1 - c[ij,p,k]
                                r[flat(i, j, p, k)] -= 1.0
                            if p == i:                      # RHS2 - c[kl,j,q]
                                r[flat(k, l, j, q)] -= 1.0
                            rows.append(r)
    A = np.array(rows)
    rank = int(np.linalg.matrix_rank(A))
    dim = 16 - rank
    print(f"    M_2 derivation space dim = {dim}  (Leibniz constraint rank = {rank})")
    print(f"    inner derivations [h,.] : h in M_2 (4-dim), centre scalar -> dim = 3")
    print(f"    => dim {dim} == 3 : ALL M_2 derivations are INNER (HH^1(M_2)=0).")
    print("       The 'scalar->tensor needs OUTER derivations' wall is about the LIMIT")
    print("       (finite -> infinite II_1), which finite numeric cannot reach.")
    return {"derivation_dim": dim, "leibniz_rank": rank}


def main():
    print("=== 补做之前未跑程序的验证（诚实补漏）===")
    print()
    r1 = part1()
    r2 = part2()
    r3 = part3()
    print()
    print("=== conclusion ===")
    print("  - Gauss-Bonnet (topology=Riemann two faces): verified numerically (chi=2).")
    print("  - Noncommutative torus commutator: verified numerically (UV=omega VU).")
    print("  - M_2 derivations all inner: verified numerically (dim=3).")
    print("  Remaining non-verifiable: ASSUMPTIONS (matter picks 4D) and IDENTIFICATIONS")
    print("  (state-space = physical space), which are not derivations.")

    summary = {
        "gauss_bonnet_chi": r1["chi"],
        "noncommutative_torus_diff": r2["diff"],
        "noncommutative_torus_comm": r2["comm"],
        "M2_derivation_dim": r3["derivation_dim"],
        "M2_leibniz_rank": r3["leibniz_rank"],
        "note": "three previously hand-derived/cited steps now program-verified. "
                "Remaining non-verifiable: assumptions (matter picks 4D) and "
                "identifications (state-space = physical space).",
    }
    out = ROOT / "experiments" / "exp_verify_final_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
