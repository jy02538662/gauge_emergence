"""时间 = 操作 = 加倍 = 观察 J；非结合 = 时间作用三次（过去/现在/未来）。

Chain to verify:
  1. 观察 J (J^2 = -1) = Cayley-Dickson imaginary unit i (i^2 = -1)  -> 时间 = 加倍.
  2. Cayley-Dickson ladder R->C->H->O = "adding imaginary units" = "时间 applied repeatedly".
  3. The octonion associator [e1,e2,e4] (= "past/now/future") is NON-zero = 非结合.

The "now" (operation) cannot be combined with "past/future" (states) -> non-associative.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def conj(x):
    n = x.shape[0]
    if n == 1:
        return x.copy()
    m = n // 2
    out = np.empty_like(x)
    out[:m] = conj(x[:m])
    out[m:] = -x[m:]
    return out


def mul(x, y):
    n = x.shape[0]
    if n == 1:
        return x * y
    m = n // 2
    a, b = x[:m], x[m:]
    c, d = y[:m], y[m:]
    ac = mul(a, c)
    db = mul(conj(d), b)
    da = mul(d, a)
    bc = mul(b, conj(c))
    return np.concatenate([ac - db, da + bc])


def assoc(x, y, z):
    return mul(mul(x, y), z) - mul(x, mul(y, z))


def main() -> None:
    # 1. 观察 J = [[0,1],[-1,0]], J^2 = -1  (theory's time/signature)
    J = np.array([[0.0, 1.0], [-1.0, 0.0]])
    J2 = J @ J
    # imaginary unit i = (0,1) in C, i^2 = -1
    i = np.array([0.0, 1.0])
    i2 = mul(i, i)

    # 2. Cayley-Dickson ladder = adding imaginary units
    dims = [1, 2, 4, 8]

    # 3. associator [e1, e2, e4] (past=e1, now=e2, future=e4) -> non-zero?
    eye = np.eye(8)
    a = assoc(eye[1], eye[2], eye[4])
    m = int(np.argmax(np.abs(a)))

    print("1. 观察 J = time direction (J^2 = -1):")
    print(f"   J^2 = {J2.tolist()}  -> -I ? {np.allclose(J2, -np.eye(2))}")
    print(f"   Cayley-Dickson imaginary unit i^2 = {i2.tolist()}  -> -1 ? {np.allclose(i2, [-1, 0])}")
    print("   => 观察 J = 时间 = Cayley-Dickson 虚单位 i（都是 J^2 = i^2 = -1）")
    print()
    print(f"2. Cayley-Dickson ladder = 时间 applied repeatedly: R({dims[0]}) -> C({dims[1]}) -> H({dims[2]}) -> O({dims[3]})")
    print("   每次加倍 = 加一个虚单位 = 加一个时间方向")
    print()
    print("3. 关联子 [过去=e1, 现在=e2, 未来=e4]:")
    print(f"   (e1 e2) e4 - e1 (e2 e4) = {a[m]:+.0f} e{m}  -> 非零? {np.linalg.norm(a) > 1e-9}")
    print("   => 「现在」（操作）无法结合进「过去/未来」（状态）= 非结合（八元数）")
    print()
    print("结论：时间=操作=加倍（观察 J=i），作用三次（过去/现在/未来）→ 非结合。")
    print("     空间=状态（可实化，A15）；时间=操作（非结合化）。统一 = 八元数。")

    out = ROOT / "experiments" / "exp_time_as_operation_last_run.json"
    out.write_text(json.dumps({
        "J_squared_minus_I": bool(np.allclose(J2, -np.eye(2))),
        "i_squared_minus_1": bool(np.allclose(i2, [-1, 0])),
        "associator_norm": float(np.linalg.norm(a)),
        "associator_nonzero": bool(np.linalg.norm(a) > 1e-9),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
