"""Re-derive step 2: WHICH curvature channel does delta live in?

Separate modulus perturbations into SCALAR (isotropic stretch s) and SHEAR
(anisotropy a) channels:
    r_x = 1 + s + a
    r_y = 1 + s - a
where s = (dr_x + dr_y)/2, a = (dr_x - dr_y)/2.

Measure delta's response to s alone and to a alone (single-frequency modes).
This determines whether the angle defect delta is the scalar curvature R
(responds to s) or a shear/Weyl-type curvature (responds to a).

Code: `py -m experiments.exp_bridge_B_rederive2`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from itertools import product

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def angle_defect_field(r_x, r_y, L):
    delta = np.zeros((L, L))

    def wrap(idx):
        return idx % L

    for p, q in product(range(L), repeat=2):
        a = r_x[p, q]
        b = r_y[wrap(p + 1), q]
        c = r_x[p, wrap(q + 1)]
        d = r_y[p, q]
        diag = np.sqrt(a**2 + b**2)

        def tri_angles(x, y, z):
            ax = np.arccos(np.clip((y**2 + z**2 - x**2) / (2 * y * z), -1, 1))
            ay = np.arccos(np.clip((x**2 + z**2 - y**2) / (2 * x * z), -1, 1))
            az = np.arccos(np.clip((x**2 + y**2 - z**2) / (2 * x * y), -1, 1))
            return ax, ay, az

        angC, angA1, angB = tri_angles(a, b, diag)
        angD, angA2, angC2 = tri_angles(diag, c, d)
        angA = angA1 + angA2
        angC_tot = angC + angC2

        delta[p, q] += angA
        delta[wrap(p + 1), q] += angB
        delta[wrap(p + 1), wrap(q + 1)] += angC_tot
        delta[p, wrap(q + 1)] += angD

    return 2 * np.pi - delta


def main():
    L = 48
    x = np.arange(L); y = np.arange(L)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    eps = 0.1

    print("=== re-derive 2: scalar channel s vs shear channel a ===")
    print(f"  L={L}, eps={eps}")
    print(f"  scalar s = (dr_x+dr_y)/2  (isotropic stretch)")
    print(f"  shear  a = (dr_x-dr_y)/2  (anisotropy)")

    # scalar channel: s = eps*cos(kx), a = 0  =>  r_x = r_y = 1 + eps*cos(kx)
    # shear  channel: a = eps*cos(kx), s = 0  =>  r_x = 1+eps*cos(kx), r_y = 1-eps*cos(kx)
    # Note BOTH have r_x-r_y structure; the KEY question is delta's ORDER in the
    # channel and its k-scaling.

    for n in [1, 2, 3, 4, 6, 8]:
        k = 2 * np.pi * n / L
        ck = np.cos(k * xx)
        # scalar (isotropic)
        rx_s = 1 + eps * ck; ry_s = 1 + eps * ck
        ds = angle_defect_field(rx_s, ry_s, L)
        # shear
        rx_a = 1 + eps * ck; ry_a = 1 - eps * ck
        da = angle_defect_field(rx_a, ry_a, L)
        print(f"  k=n{n}: scalar max|delta|={np.max(np.abs(ds)):.3e},  "
              f"shear max|delta|={np.max(np.abs(da)):.3e}")

    print()
    print("  Note: scalar(s) and shear(a) give the SAME delta here because")
    print("  a single-frequency cos(kx) is a pure-stretch in both; the real")
    print("  discriminator is a=cos(kx), s=cos(ky) (cross) vs s=a=cos(kx).")
    print()

    # Better discriminator: which channel carries the k^2 (Laplacian) behavior?
    # scalar s = cos(kx)cos(ky) (Laplacian-like), shear a = cos(kx)-cos(ky).
    k1 = 2 * np.pi * 3 / L
    k2 = 2 * np.pi * 4 / L
    ck1x = np.cos(k1 * xx); ck2y = np.cos(k2 * yy)
    # scalar: isotropic, product (grad^2-like)
    rx = 1 + eps * ck1x * ck2y; ry = 1 + eps * ck1x * ck2y
    d_scalar = angle_defect_field(rx, ry, L)
    # shear: difference
    rx2 = 1 + eps * (ck1x - ck2y); ry2 = 1 - eps * (ck1x - ck2y)
    d_shear = angle_defect_field(rx2, ry2, L)
    print(f"  scalar s=cos(k1 x)cos(k2 y): max|delta| = {np.max(np.abs(d_scalar)):.3e}")
    print(f"  shear  a=cos(k1 x)-cos(k2 y): max|delta| = {np.max(np.abs(d_shear)):.3e}")

    summary = {
        "L": L, "eps": eps,
        "note": "single-freq scalar and shear give same delta (both pure stretch). "
                "The channel separation needs a cross/product pattern; see the two "
                "discriminator lines at the end.",
        "scalar_product_maxdelta": float(np.max(np.abs(d_scalar))),
        "shear_diff_maxdelta": float(np.max(np.abs(d_shear))),
    }
    out = ROOT / "experiments" / "exp_bridge_B_rederive2_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
