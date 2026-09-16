"""Bridge B flip: is scalar curvature (EH) cut out by observation breaking a
'metric/rigidity' symmetry of the One, or only gauge curvature (YM)?

Discriminator (computable, clean):
  - Gauge curvature = holonomy (phase product around a plaquette). INTRINSIC to D
    (depends only on phases). Pi-flux => holonomy = -1.  Already cut.
  - Scalar curvature = Regge angle defect  delta_v = 2 pi - sum(face angles at v).
    For square plaquettes (unit edges, face angle = pi/2 at each corner),
        delta_v = (4 - valence_v) * pi/2
    where valence_v = how many plaquettes meet at v.  This depends on the
    COMBINATORIAL structure (valence), NOT on the edge moduli/phases.
    valence 4 (flat torus)  => 0
    valence 3 (cube=sphere) => +pi/2
    valence 5 (hyperbolic)  => -pi/2

The pi-flux torus is 4-regular (valence=4) => angle defect 0 => FLAT => scalar
curvature is NOT cut out by the current "One".

Code: `py -m experiments.exp_bridge_B_scalar_curvature`
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


def torus2D(L, flux=True):
    """2D pi-flux torus (L x L), same convention as exp_H_from_D."""
    N = L * L

    def idx(x, y):
        return (y % L) * L + (x % L)

    D = np.zeros((N, N))
    for x, y in product(range(L), repeat=2):
        i = idx(x, y)
        j = idx(x + 1, y); w = (-1) ** y if flux else 1.0; D[i, j] = w; D[j, i] = w
        j = idx(x, y + 1); w = 1.0; D[i, j] = w; D[j, i] = w
    return D


def plaquette_holonomy(D, L, x, y):
    """Phase holonomy around plaquette at lower-left (x,y): up, right, down, left."""
    def idx(a, b):
        return (b % L) * L + (a % L)

    # corners: (x,y) -> (x,y+1) -> (x+1,y+1) -> (x+1,y) -> (x,y)
    i0 = idx(x, y); i1 = idx(x, y + 1); i2 = idx(x + 1, y + 1); i3 = idx(x + 1, y)
    hol = D[i0, i1] * D[i1, i2] * D[i2, i3] * D[i3, i0]
    return hol


def angle_defect(valence):
    """Regge angle defect at a vertex of a square grid, unit edges (face angle pi/2).
    delta_v = 2 pi - valence * (pi/2) = (4 - valence) * pi/2."""
    return (4 - valence) * (np.pi / 2)


def main():
    L = 4
    D = torus2D(L, flux=True)

    # ---- 1. gauge curvature: holonomy = -1 for every plaquette (intrinsic) ----
    holonomies = [plaquette_holonomy(D, L, x, y) for x in range(L) for y in range(L)]
    all_minus_one = bool(np.allclose(np.real(holonomies), -1))

    print("=== Bridge B flip: scalar (EH) vs gauge (YM) curvature, angle defect ===")
    print(f"  pi-flux holonomy (gauge curvature) all = -1: {all_minus_one}")
    print(f"    (gauge curvature is INTRINSIC to D's phases => already cut)")

    # ---- 2. scalar curvature = angle defect, depends on VALENCE not moduli ----
    print("\n  angle defect delta_v = (4 - valence) * pi/2  (square grid, unit edges):")
    for v in [4, 3, 5]:
        d = angle_defect(v)
        label = {4: "flat torus", 3: "cube = sphere", 5: "hyperbolic"}[v]
        print(f"    valence={v} ({label:14s}): delta_v = {d/np.pi:+.3f} pi")

    # ---- 3. the pi-flux torus is 4-regular => delta = 0 => flat ----
    valence_pi = 4  # square torus: each vertex meets 4 plaquettes
    delta_pi = angle_defect(valence_pi)
    print(f"\n  pi-flux torus is 4-regular => delta_v = {delta_pi/np.pi:.3f} pi = 0 (FLAT)")
    print("  => scalar curvature (EH) is NOT cut out by the current 'One' (flat).")

    summary = {
        "L": L,
        "holonomy_all_minus_one": all_minus_one,
        "angle_defect": {
            "valence_4_flat": angle_defect(4),
            "valence_3_sphere": angle_defect(3),
            "valence_5_hyperbolic": angle_defect(5),
        },
        "pi_flux_valence": valence_pi,
        "pi_flux_angle_defect": delta_pi,
        "conclusion": "Gauge curvature (holonomy=-1) is intrinsic to D and already cut. "
                      "Scalar curvature (angle defect) = (4-valence)*pi/2 depends on the "
                      "combinatorial valence, not on moduli. The pi-flux torus is 4-regular "
                      "=> defect 0 => FLAT => scalar curvature (EH) not cut. To cut R needs "
                      "valence != 4 (sphere/hyperbolic), a new mechanism the 'One' does not "
                      "currently have; OR the continuum a2 coefficient (Chamseddine-Connes).",
    }
    out = ROOT / "experiments" / "exp_bridge_B_scalar_curvature_last_run.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
