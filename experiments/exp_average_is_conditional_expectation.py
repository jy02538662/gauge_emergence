"""检验：弱交换化实验的"平均" = 条件期望 E（观察）？

四性质：幂等 / 保迹 / 正 / 双模。
四种候选平均：M1 部分迹 / M2 块对角 / M3 模流时间平均 / M4 平移平均。
关键问题：哪种像代数交换（= 交换化，P1 裂缝），M3 vs M4 是否相等（时间 vs 空间观察）。

Code: `py -m experiments.exp_average_is_conditional_expectation`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np.random.seed(42)


def build_system(n):
    N = 2**n
    lambdas = np.array([1.0 / (i + 1) for i in range(N)])
    T = np.zeros((N, N))
    for i in range(N):
        T[(i + 1) % N, i] = 1.0
    return N, lambdas, T


# ---------- 四种候选"平均" ----------
def M1(x):  # 部分迹（对第二个 M_2 因子取迹，扩展回 N x N）
    N = x.shape[0]
    h = N // 2
    e = (x[:h, :h] + x[h:, h:]) / 2
    r = np.zeros_like(x, dtype=complex)
    r[:h, :h] = e
    r[h:, h:] = e
    return r


def M2(x):  # 块对角
    N = x.shape[0]
    h = N // 2
    r = np.zeros_like(x, dtype=complex)
    r[:h, :h] = x[:h, :h]
    r[h:, h:] = x[h:, h:]
    return r


def M3(x, lambdas):  # 模流时间平均（投影到 {rho}'，rho 谱互异 => 对角）
    N = x.shape[0]
    r = np.zeros_like(x, dtype=complex)
    for i in range(N):
        for j in range(N):
            if abs(lambdas[i] - lambdas[j]) < 1e-12:
                r[i, j] = x[i, j]
    return r


def M4(x, T):  # 平移平均（投影到平移不变子代数 = 循环矩阵）
    N = x.shape[0]
    Tinv = T.T
    r = np.zeros_like(x, dtype=complex)
    Tk = np.eye(N, dtype=complex)
    Tinvk = np.eye(N, dtype=complex)
    for k in range(N):
        r += Tk @ x @ Tinvk
        Tk = Tk @ T
        Tinvk = Tinvk @ Tinv
    return r / N


# ---------- 条件期望四性质 ----------
def ce_props(M, x, x_psd):
    idem = np.linalg.norm(M(M(x)) - M(x))                      # 幂等
    trace = abs(np.trace(M(x)) - np.trace(x))                  # 保迹
    pos = np.min(np.linalg.eigvalsh((M(x_psd) + M(x_psd).conj().T) / 2))  # 正
    d = M(x)
    bimod = np.linalg.norm(M(d @ x @ d) - d @ M(x) @ d)        # 双模（d=M(x) 情形）
    return idem, trace, pos, bimod


def main():
    print("=" * 78)
    print("检验：'平均' 是否 = 条件期望 E（观察）？")
    print("=" * 78)

    ce_table = {}
    for n in [3, 4, 5]:
        N, lambdas, T = build_system(n)
        A = np.random.randn(N, N) + 1j * np.random.randn(N, N)
        A = (A + A.conj().T) / 2
        B = np.random.randn(N, N) + 1j * np.random.randn(N, N)
        Bpsd = B @ B.conj().T

        print(f"\nN = {N}")
        print(f"{'平均':<12} {'幂等':<10} {'保迹':<10} {'正':<12} {'双模':<10}")
        print("-" * 78)
        candidates = [
            ("M1 部分迹", lambda x: M1(x)),
            ("M2 块对角", lambda x: M2(x)),
            ("M3 模流", lambda x: M3(x, lambdas)),
            ("M4 平移", lambda x: M4(x, T)),
        ]
        for name, M in candidates:
            idem, trace, pos, bimod = ce_props(M, A, Bpsd)
            print(f"{name:<12} {idem:<10.2e} {trace:<10.2e} {pos:<12.2e} {bimod:<10.2e}")
            ce_table[f"{name}_N{N}"] = {"idem": idem, "trace": trace, "pos": pos, "bimod": bimod}

    # ---------- 像代数交换性 ----------
    print("\n" + "=" * 78)
    print("像代数交换性 ||[a,b]||, a,b 在像里（P1 裂缝的数值签名）")
    print("=" * 78)
    comm_table = {}
    for n in [3, 4, 5, 6]:
        N, lambdas, T = build_system(n)
        print(f"\nN = {N}")
        for name, M in [
            ("M1", lambda x: M1(x)),
            ("M2", lambda x: M2(x)),
            ("M3", lambda x: M3(x, lambdas)),
            ("M4", lambda x: M4(x, T)),
        ]:
            a = M(np.random.randn(N, N) + 1j * np.random.randn(N, N))
            b = M(np.random.randn(N, N) + 1j * np.random.randn(N, N))
            comm = np.linalg.norm(a @ b - b @ a)
            print(f"  {name}: ||[a,b]|| = {comm:.4e}")
            comm_table[f"{name}_N{N}"] = comm

    # ---------- M3 vs M4 ----------
    print("\n" + "=" * 78)
    print("M3（模流） vs M4（平移）")
    print("=" * 78)
    n = 5
    N, lambdas, T = build_system(n)
    A = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    M3A = M3(A, lambdas)
    M4A = M4(A, T)
    diff = np.linalg.norm(M3A - M4A)
    print(f"||M3(A) - M4(A)|| = {diff:.4e}")
    rho = np.diag(lambdas)
    T_rho_Tinv = T @ rho @ T.T
    rho_diff = np.linalg.norm(T_rho_Tinv - rho)
    print(f"||T rho T+ - rho|| = {rho_diff:.4e}")
    if rho_diff < 1e-10:
        print("-> rho 平移不变：M3 和 M4 相容")
    else:
        print("-> rho 不平移不变：M3 != M4（投影到不同子代数）")

    summary = {
        "question": "is 'averaging' = conditional expectation E (observation)?",
        "four_props_all_pass": "see ce_table (expect all ~0 / pos>=0)",
        "image_commutativity": comm_table,
        "M3_vs_M4_diff": float(diff),
        "rho_not_translation_invariant": bool(rho_diff > 1e-10),
        "conclusion_hint": "M3 image = diagonal (abelian, rho spectrum distinct), "
                           "M4 image = circulant (abelian). BOTH abelian but DIFFERENT. "
                           "M3 != M4 since rho = diag(1/i) not translation-invariant.",
    }
    out = ROOT / "experiments" / "exp_average_is_conditional_expectation_last_run.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
