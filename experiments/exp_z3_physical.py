"""Is omega (the Z3 3-cycle) a PHYSICAL operator, or only an abstract automorphism?

User's fork: on 4x4 (chirality Gamma x conjugation K), the 3-cycle permutes
{Gamma, K, Gamma K}, but Gamma is UNITARY and K is ANTI-UNITARY, so a physical
(anti)unitary conjugation cannot mix them -> omega is NOT physical there (expected fail).

On 8-dim (octonions / Spin(8)), triality permutes three UNITARY reps (all same type),
so the S3 (containing Z3) IS physical (expected success).

Part 1 (4x4): type mismatch -> no physical 3-cycle.
Part 2 (8-dim): find a UNITARY order-3 automorphism of the octonions (Z3 in G2).
"""

from __future__ import annotations

import json
from itertools import permutations
from math import gcd
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def part1():
    # Gamma: unitary (diagonal +-1, Gamma^2 = 1); K: anti-unitary (complex conjugation).
    # The 3-cycle needs: Gamma (unitary) -> K (anti-unitary)  => type FLIPPED => needs anti-unitary U
    #                      K (anti-unitary) -> Gamma K (anti-unitary) => type PRESERVED => needs unitary U
    # Contradiction -> no single physical U realizes the 3-cycle.
    types = {"Gamma": "unitary", "K": "anti-unitary", "GammaK": "anti-unitary"}
    cycle = {"Gamma": "K", "K": "GammaK", "GammaK": "Gamma"}
    # check: for each pair, does the type flip or preserve?
    flip_required = []
    for src, dst in cycle.items():
        same = types[src] == types[dst]
        flip_required.append((src, dst, "preserve" if same else "flip"))
    # a unitary U preserves type; an anti-unitary U flips type.
    # The cycle has BOTH preserve and flip -> impossible for a single U.
    needs = {src: ("unitary" if types[src] == types[dst] else "anti-unitary") for src, dst in cycle.items()}
    consistent = len(set(needs.values())) == 1
    print("Part 1 (4x4): the 3-cycle {Gamma -> K -> GammaK -> Gamma}")
    print("  types:", types)
    for src, dst, f in flip_required:
        print(f"    {src} ({types[src]}) -> {dst} ({types[dst]}): type {f}")
    print(f"  required U type per map: {needs}")
    print(f"  => single physical U possible? {consistent}   (expected False)")
    return {"physical_on_4x4": bool(consistent)}


def part2():
    # Fano plane lines of the octonion imaginary units (from exp_associator_antisym).
    lines = [{1, 2, 3}, {1, 4, 5}, {1, 6, 7}, {2, 4, 6}, {2, 5, 7}, {3, 4, 7}, {3, 5, 6}]
    lineset = {frozenset(l) for l in lines}

    def order3(pi):
        # order of permutation pi (tuple mapping 1..7 -> 1..7)
        seen = [False] * 8
        o = 1
        for i in range(1, 8):
            if not seen[i]:
                cyc = 0
                j = i
                while not seen[j]:
                    seen[j] = True
                    cyc += 1
                    j = pi[j - 1]
                o = o * cyc // gcd(o, cyc)
        return o

    found = []
    for pi in permutations(range(1, 8)):
        mapped = {frozenset(pi[x - 1] for x in l) for l in lines}
        if mapped == lineset:
            o = order3(pi)
            if o == 3:
                found.append(pi)
    print("\nPart 2 (8-dim): unitary order-3 automorphisms of the octonion (Z3 in G2)")
    print(f"  # Fano-plane automorphisms of order 3 = {len(found)}")
    if found:
        pi = found[0]
        print(f"  example 3-cycle: {pi}")
        print(f"  => Z3 IS realized as a UNITARY (orthogonal) automorphism on 8-dim: physical")
    return {"n_order3_automorphisms": len(found), "example": list(found[0]) if found else None}


def main() -> None:
    r1 = part1()
    r2 = part2()
    out = ROOT / "experiments" / "exp_z3_physical_last_run.json"
    out.write_text(json.dumps({**r1, **r2}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
