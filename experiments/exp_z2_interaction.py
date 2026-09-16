"""Two independent Z2's -> Aut(Z2 x Z2) = S3 -> Z3 (the source of omega).

User's proposal: derive omega = e^{2 pi i / 3} from the interaction of two independent
Z2's, since Aut(Z2 x Z2) = S3 (permutations of the 3 non-trivial elements), and S3
contains a Z3 (the 3-cycle).

Part 1 (pure math): verify Aut(Z2 x Z2) = S3 and exhibit the Z3 (3-cycle).
Part 2 (theory check): build the pi-flux D (real gauge), find its two Z2's --
  chirality Gamma (bipartite) and conjugation K (self-reflexivity) -- and check
  whether they COMMUTE (giving Z2 x Z2), i.e. whether the theory has two independent Z2's.
"""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# ---------- Part 1: Aut(Z2 x Z2) = S3 ----------

def z2z2_group():
    # Z2 x Z2, elements as tuples; identity (0,0); non-trivial: (0,1),(1,0),(1,1)
    return [(0, 0), (0, 1), (1, 0), (1, 1)]


def add(a, b):
    return ((a[0] + b[0]) % 2, (a[1] + b[1]) % 2)


def part1():
    elems = z2z2_group()
    nontrivial = [(0, 1), (1, 0), (1, 1)]  # a, b, ab
    valid_auts = 0
    for perm in permutations(nontrivial):
        # map identity -> identity, nontrivial[i] -> perm[i]
        m = {(0, 0): (0, 0)}
        for i, e in enumerate(nontrivial):
            m[e] = perm[i]
        # check homomorphism: m(x+y) == m(x)+m(y) for all x,y
        ok = all(m[add(x, y)] == add(m[x], m[y]) for x in elems for y in elems)
        if ok:
            valid_auts += 1
    # the 3-cycle (a->b->c->a) generates Z3
    three_cycle = {(0, 1): (1, 0), (1, 0): (1, 1), (1, 1): (0, 1), (0, 0): (0, 0)}
    # powers of the 3-cycle: id, (abc), (acb)  => order 3
    print("Part 1: Aut(Z2 x Z2)")
    print(f"  # valid automorphisms = {valid_auts}  (S3 has 6)")
    print(f"  the 3-cycle (a->b->c->a) has order 3 => Z3 subgroup exists: {valid_auts == 6}")
    return {"n_automorphisms": valid_auts, "is_S3": valid_auts == 6}


# ---------- Part 2: pi-flux D has two commuting Z2's ----------

def build_pi_flux(L):
    """LxL torus, pi flux per plaquette, real gauge (horizontal +1, vertical +1/-1 by x parity)."""
    n = L * L
    D = np.zeros((n, n))
    for x in range(L):
        for y in range(L):
            i = x + L * y
            # horizontal +1
            D[i, (x + 1) % L + L * y] += 1.0
            # vertical: +1 if x even, -1 if x odd
            D[i, x + L * ((y + 1) % L)] += 1.0 if x % 2 == 0 else -1.0
    D = (D + D.T) / 2.0  # symmetrize (undirected Hermitian real)
    return D


def part2(L=4):
    D = build_pi_flux(L)
    n = D.shape[0]
    # chirality Gamma: bipartite (-1)^(x+y)
    gamma = np.zeros((n, n))
    for x in range(L):
        for y in range(L):
            i = x + L * y
            gamma[i, i] = 1.0 if (x + y) % 2 == 0 else -1.0
    # check Gamma D Gamma = -D (chirality)
    chirality_ok = np.allclose(gamma @ D @ gamma, -D, atol=1e-9)
    gamma2_ok = np.allclose(gamma @ gamma, np.eye(n), atol=1e-9)
    # conjugation K: D is real => K D K = D, and K Gamma K = Gamma (Gamma real)
    D_real = np.allclose(D, D.real, atol=1e-12) and np.max(np.abs(D.imag)) < 1e-12
    # Gamma and K commute iff Gamma is real (it is, diagonal +-1)
    commute_ok = np.allclose(gamma, gamma.conj(), atol=1e-12)

    print("\nPart 2: pi-flux D (L=%d)" % L)
    print(f"  chirality Gamma: Gamma D Gamma = -D ? {chirality_ok}   Gamma^2 = I ? {gamma2_ok}")
    print(f"  D real (so conjugation K is a symmetry) ? {D_real}")
    print(f"  Gamma real (so Gamma and K commute) ? {commute_ok}")
    print(f"  => {{1, Gamma, K, Gamma K}} = Z2 x Z2 (two independent Z2's) ? {chirality_ok and gamma2_ok and D_real and commute_ok}")

    return {
        "chirality_ok": bool(chirality_ok),
        "gamma2_ok": bool(gamma2_ok),
        "D_real": bool(D_real),
        "gamma_real_commute": bool(commute_ok),
        "has_Z2xZ2": bool(chirality_ok and gamma2_ok and D_real and commute_ok),
    }


def main() -> None:
    r1 = part1()
    print()
    r2 = part2(4)
    out = ROOT / "experiments" / "exp_z2_interaction_last_run.json"
    out.write_text(json.dumps({**r1, **r2}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
