"""Verify the octonion associator [x,y,z] = (xy)z - x(yz) is totally antisymmetric,
and non-zero only for "non-collinear" triples (Fano plane structure).

Entry 1 (三阶自反) first stepping stone: is "total antisymmetry of the associator"
(3-fold self-conjugacy) the characterizing property of the octonion (non-assoc) structure?
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


def assoc(x, y, z):
    return mul_rec(mul_rec(x, y), z) - mul_rec(x, mul_rec(y, z))


def main() -> None:
    dim = 8
    eye = np.eye(dim)

    nz_triples = []
    antisym_ok = True
    cyclic_ok = True
    for i in range(1, 8):
        for j in range(1, 8):
            for k in range(1, 8):
                a = assoc(eye[i], eye[j], eye[k])
                if np.linalg.norm(a) > 1e-10:
                    nz_triples.append((i, j, k, a))

    for (i, j, k, a) in nz_triples:
        if not np.allclose(a, -assoc(eye[j], eye[i], eye[k]), atol=1e-9):
            antisym_ok = False
        if not np.allclose(a, -assoc(eye[i], eye[k], eye[j]), atol=1e-9):
            antisym_ok = False
        if not np.allclose(a, assoc(eye[j], eye[k], eye[i]), atol=1e-9):
            cyclic_ok = False

    lines = []
    for i in range(1, 8):
        for j in range(i + 1, 8):
            for k in range(j + 1, 8):
                if np.linalg.norm(assoc(eye[i], eye[j], eye[k])) < 1e-10:
                    lines.append([i, j, k])

    print("Octonion associator over imaginary units e1..e7:")
    print(f"  # non-zero triples = {len(nz_triples)}  (expected 168 = 28*6)")
    print(f"  total antisymmetry (swap flips sign) = {antisym_ok}")
    print(f"  cyclic invariance = {cyclic_ok}")
    print(f"  # zero-associator 'lines' (distinct i<j<k) = {len(lines)}  (expected 7)")
    print(f"  lines (Fano plane): {lines}")

    if nz_triples:
        i, j, k, a = nz_triples[0]
        m = int(np.argmax(np.abs(a)))
        print(f"\n  example: [e{i},e{j},e{k}] -> e{m}  sign {a[m]:+.0f}")

    out = ROOT / "experiments" / "exp_associator_antisym_last_run.json"
    out.write_text(json.dumps({
        "n_nonzero_triples": len(nz_triples),
        "total_antisymmetry": bool(antisym_ok),
        "cyclic_invariance": bool(cyclic_ok),
        "n_lines": len(lines),
        "lines": lines,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
