"""Binary relation (k=ij) vs ternary associator: does non-associativity emerge from
"relations within H", or does it REQUIRE the new unit e4?

命题（2026-09-21，方向 C「因果 = 三元」收窄后的检验）：
  「第三次加倍 = 关系本身的涌现（因果）」——若成立，则二元关系 k = e1*e2（在 H 内）
  应已经给出非结合。本脚本用 Cayley–Dickson 乘法验证四件事，把这句钉死：
    1. H（四元数，4 维）内所有关联子 = 0  -> H 结合；
    2. 「关系」k = e1*e2 仍在 H 内，[e1,e2,k] = 0 -> 二元关系结合，给不了非结合；
    3. O（八元数，8 维）[e1,e2,e4] = 2e7 != 0 -> 非结合必须新单位 e4 参与；
    4. O 内只用 {e1,e2,e3}（H 子代数）的关联子 = 0 -> 没有 e4 就没有非结合。

结论：非结合性是「引入 e4 之后的后果」，不是「关系本身的涌现」。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def conj(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    if n == 1:
        return x.copy()
    m = n // 2
    out = np.empty_like(x)
    out[:m] = conj(x[:m])
    out[m:] = -x[m:]
    return out


def mul_rec(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cayley–Dickson multiplication (recursive)."""
    n = x.shape[0]
    if n == 1:
        return x * y
    m = n // 2
    a, b = x[:m], x[m:]
    c, d = y[:m], y[m:]
    ac = mul_rec(a, c)
    db = mul_rec(conj(d), b)
    da = mul_rec(d, a)
    bc = mul_rec(b, conj(c))
    return np.concatenate([ac - db, da + bc])


def assoc(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return mul_rec(mul_rec(x, y), z) - mul_rec(x, mul_rec(y, z))


def basis(dim: int, k: int) -> np.ndarray:
    return np.eye(dim)[k]


def main() -> None:
    # 1) H (dim 4): all associators within H are zero -> associative
    h_assoc_zero = True
    h_triples = [(1, 2, 3), (1, 3, 2), (2, 3, 1), (2, 1, 3), (3, 1, 2), (3, 2, 1)]
    for (i, j, k) in h_triples:
        if np.linalg.norm(assoc(basis(4, i), basis(4, j), basis(4, k))) > 1e-9:
            h_assoc_zero = False

    # 2) the "relation" k = e1*e2 lives inside H, still associative
    e1h, e2h = basis(4, 1), basis(4, 2)
    k12 = mul_rec(e1h, e2h)
    rel_assoc = assoc(e1h, e2h, k12)
    rel_is_zero = bool(np.linalg.norm(rel_assoc) < 1e-9)

    # 3) O (dim 8): [e1,e2,e4] != 0 -> non-assoc REQUIRES e4
    E1, E2, E4 = basis(8, 1), basis(8, 2), basis(8, 4)
    a124 = assoc(E1, E2, E4)
    a124_nonzero = bool(np.linalg.norm(a124) > 1e-9)
    a124_unit = int(np.argmax(np.abs(a124)))
    a124_sign = float(a124[a124_unit])

    # 4) inside O, associators using ONLY {e1,e2,e3} (the H subalgebra) stay zero
    o_h_zero = True
    for (i, j, k) in [(1, 2, 3), (2, 3, 1), (3, 1, 2)]:
        if np.linalg.norm(assoc(basis(8, i), basis(8, j), basis(8, k))) > 1e-9:
            o_h_zero = False

    print("Binary relation vs ternary associator (非结合是否 = 「关系本身」的涌现):")
    print(f"  1) H (dim 4) all associators zero (associative) = {h_assoc_zero}")
    print(f"  2) relation k=e1*e2 inside H: [e1,e2,k] zero = {rel_is_zero}  (binary relation -> associative)")
    print(f"  3) O (dim 8): [e1,e2,e4] nonzero = {a124_nonzero}  (unit e{a124_unit}, sign {a124_sign:+.0f})  -> needs NEW unit e4")
    print(f"  4) O using only {{e1,e2,e3}} (H-subalgebra) zero = {o_h_zero}")
    print("  => non-assoc is the CONSEQUENCE of introducing e4, NOT the emergence of a 'relation'.")

    out = ROOT / "experiments" / "exp_relation_vs_associator_last_run.json"
    out.write_text(json.dumps({
        "H_associative": h_assoc_zero,
        "binary_relation_k_associative": rel_is_zero,
        "octonion_associator_e1e2e4_nonzero": a124_nonzero,
        "octonion_associator_e1e2e4_unit": a124_unit,
        "octonion_associator_e1e2e4_sign": a124_sign,
        "octonion_H_subalgebra_associative": o_h_zero,
        "conclusion": "non-associativity requires the NEW unit e4, not a relation within H",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
