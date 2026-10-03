import sympy as sp
from sympy import symbols, pi, integrate, exp, oo, Rational, simplify

# --- 1. 谱作用量耦合常数匹配（符号） ---
f2, f4, L = symbols('f2 f4 Lambda', positive=True)
G, g = symbols('G g', positive=True)

# 8πG 匹配（预印本1.13）：f2*Lambda^2 * a2 = (1/16πG)∫R, a2=-R/(48π^2) (4D Dirac)
G_expr = 3*pi/(f2*L**2)
print('G =', G_expr)

# YM 匹配：a4 ⊃ -(1/24π^2)∫tr F^2, 谱作用量 a4 阶系数 f4 (Lambda^0)
# f4/(24π^2) = 1/(4g^2)  ->  g^2 = 6π^2/f4
g2_expr = 6*pi**2/f4
print('g^2 =', g2_expr)

print('G*Lambda^2 =', simplify(G_expr*L**2), '  (只依赖 f2, 无量纲)')
print('G/g^2 =', simplify(G_expr/g2_expr), '  (含 Lambda^2 + f4/f2)')

# --- 2. f 的矩比值 ---
# 假设 f(u) = (1/u)*exp(-u/u0)，u = D^2/Lambda^2，u0 是截断（对应观察者 λ_c / Λ^2）
u0 = symbols('u0', positive=True)
u = symbols('u', positive=True)
f = (1/u)*exp(-u/u0)

# 矩的定义：f_k = ∫ f(u) u^k du  (k 对应 Λ^{2-2k} 阶？需要仔细)
# 标准 Chamseddine-Connes: S = Σ f_{2n} a_{2n},  f_{2n} = ∫ f(u) u^{n-1} du (u=D^2/Λ^2)
# f_0 (Λ^4): ∫ f(u) u^{-1} du ; f_2 (Λ^2): ∫ f(u) u^{0} du ; f_4 (Λ^0): ∫ f(u) u^{+1} du
f0 = integrate(f * u**(-1), (u, 0, oo))  # Λ^4 阶
f2m = integrate(f * u**(0), (u, 0, oo))  # Λ^2 阶
f4m = integrate(f * u**(+1), (u, 0, oo)) # Λ^0 阶
print('\n[标准矩 f_0, f_2, f_4 (u=D^2/Λ^2)]')
print('f_0 (Λ^4) =', f0)
print('f_2 (Λ^2) =', f2m)
print('f_4 (Λ^0) =', f4m)

# 无量纲比值（u0 是否消掉？）
print('\n[矩比值，看 u0 是否消掉]')
print('f_4/f_2 =', simplify(f4m/f2m))
print('f_2/f_0 =', simplify(f2m/f0))
print('f_4/f_0 =', simplify(f4m/f0))
