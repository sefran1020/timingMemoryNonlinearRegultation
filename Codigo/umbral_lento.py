"""Umbral garantizado del teorema de ajuste lento para el ejemplo racional.

Para el subnivel Omega_l = {W <= l}, con W = V + Psi, se acotan las cinco
constantes de la prueba y se evalúa
    kappa_bar = 4 c_- p0 / (4 c_- k1 k2 + (k1 + c_+ k2)^2).
Todas las cotas son explícitas (monotonía o subdivisión de intervalos con extremos
racionales); los logaritmos y las raíces de t - 1 - ln t = c se calculan con 50
dígitos y se ensanchan hacia afuera en 1e-30. Para la respuesta racional
g(G) = Z G/(h+G) se usa la identidad exacta
    -F(z) = (z - z*) Phi(z),  Phi(z) = 1 + |G'| Z h / ((h + G(z)) (h + G(z*))),
de modo que Psi'/(-F) = 1/((zeta - z) Phi(z)).

Uso: python Codigo/umbral_lento.py   Salida: Resultados/limites-globales/umbral-lento.json
"""
import json
from fractions import Fraction as Fr
from pathlib import Path

import mpmath as mp

mp.mp.dps = 50
RAIZ = Path(__file__).resolve().parents[1]
MARGEN = mp.mpf("1e-30")

# ejemplo racional (mpf exactos: todos los parámetros son enteros)
M = lambda v: mp.mpf(v)
r, q, a, b, m, c, e, d = map(M, (12, 1, 1, 10, 2, 4, 1, 1))
th, ph, h, Z = M(1), M(8), M(81), M(10)
D = q * e + a * c
X0, Y0 = (e * r + a * m) / D, (c * r - q * m) / D
A, B0 = (a * d - b * e) / D, (c * b + q * d) / D
zeta = (c * r - q * m) / (c * b + q * d)
zs = M(1)
Gp = th * A - ph * B0                       # G'(z) < 0
G = lambda z: th * (X0 + A * z) + ph * (Y0 - B0 * z)
xh = lambda z: X0 + A * z
yh = lambda z: Y0 - B0 * z


def Psi(z):
    z = mp.mpf(z)
    return -(z - zs) - (zeta - zs) * mp.log((zeta - z) / (zeta - zs))


def raices_h(cte):
    """t_- < 1 < t_+ con t - 1 - ln t = cte."""
    # en u = ln t: e^u - 1 - u = cte; raíz negativa en [-cte-1, 0], positiva en [0, cte+2]
    fu = lambda u: mp.exp(u) - 1 - u - cte

    def biseccion(lo, hi):
        for _ in range(400):
            mid = (lo + hi) / 2
            if (fu(lo) > 0) == (fu(mid) > 0):
                lo = mid
            else:
                hi = mid
        return lo, hi
    lo, hi = biseccion(-cte - 1, mp.mpf(0))
    tm = mp.exp(lo)                      # extremo izquierdo: cota inferior de t_-
    lo2, hi2 = biseccion(mp.mpf(0), cte + 2)
    tp = mp.exp(hi2)
    return tm - MARGEN * tm, tp + MARGEN


def cotas(ell):
    ell = mp.mpf(ell)
    # rango de z: Psi <= ell (Psi convexa, mínimo 0 en z*)
    fz = lambda z: Psi(z) - ell
    def bis(f, lo, hi):
        for _ in range(400):
            mid = (lo + hi) / 2
            if (f(lo) > 0) == (f(mid) > 0):
                lo = mid
            else:
                hi = mid
        return lo, hi
    z_hi = bis(fz, mp.mpf(zs), mp.mpf(zeta) - mp.mpf("1e-45"))[1] + MARGEN
    z_lo = mp.mpf(0) if Psi(0) <= ell else bis(fz, mp.mpf(0), mp.mpf(zs))[0] - MARGEN
    # mínimos de x, y, x^, y^ en Omega_l: V <= ell => c x^ h(x/x^) <= ell y a y^ h(y/y^) <= ell
    # x^ decrece en z (A<0) y y^ decrece en z; t_-(cte) decrece al crecer cte, así que el peor caso
    # está en z_hi (x^ e y^ mínimos): x >= x^(z_hi) t_-(ell/(c x^(z_hi))).
    # Cota refinada: en z fijo, V <= ell - Psi(z). Por subintervalos [z1, z2]:
    # x^, y^ >= su valor en z2 (decrecientes) y el presupuesto <= ell - min Psi en [z1, z2].
    assert xh(z_hi) > 0 and yh(z_hi) > 0
    zmin_psi = zs
    mx, my = mp.inf, mp.inf
    NZ = 2000
    for i in range(NZ):
        z1 = z_lo + (z_hi - z_lo) * i / NZ
        z2 = z_lo + (z_hi - z_lo) * (i + 1) / NZ
        psimin = mp.mpf(0) if z1 <= zmin_psi <= z2 else min(Psi(z1), Psi(z2))
        pres = max(ell - psimin, mp.mpf(0)) + MARGEN
        xa, ya = xh(z2), yh(z2)
        tmx, _ = raices_h(pres / (c * xa))
        tmy, _ = raices_h(pres / (a * ya))
        mx = min(mx, xa * tmx)
        my = min(my, ya * tmy)
    p0 = min(c * q, a * e)
    k1 = mp.sqrt((a * B0 / my) ** 2 + (c * abs(A) / mx) ** 2)
    k2 = (Z / h) * mp.sqrt(th ** 2 + ph ** 2)       # |g'| <= Z/h para g = Z G/(h+G), G >= 0
    # c_-, c_+: extremos de R(z) = 1/((zeta - z) Phi(z)) en [z_lo, z_hi], por subdivisión
    Gs = G(zs)
    Phi = lambda z: 1 + abs(Gp) * Z * h / ((h + (mp.mpf(Gs) + Gp * (z - zs))) * (h + Gs))
    N = 4000
    cmin, cmax = mp.inf, mp.mpf(0)
    for i in range(N):
        z1 = z_lo + (z_hi - z_lo) * i / N
        z2 = z_lo + (z_hi - z_lo) * (i + 1) / N
        # (zeta - z) en [zeta - z2, zeta - z1]; Phi crece en z (G decrece): Phi en [Phi(z1), Phi(z2)]
        lo = 1 / ((zeta - z1) * Phi(z2))
        hi = 1 / ((zeta - z2) * Phi(z1))
        cmin, cmax = min(cmin, lo), max(cmax, hi)
    kbar = 4 * cmin * p0 / (4 * cmin * k1 * k2 + (k1 + cmax * k2) ** 2)
    return dict(ell=float(ell), z_rango=[mp.nstr(z_lo, 12), mp.nstr(z_hi, 12)],
                x_min=mp.nstr(mx, 12), y_min=mp.nstr(my, 12), p0=mp.nstr(p0, 5),
                k1=mp.nstr(k1, 12), k2=mp.nstr(k2, 12), c_menos=mp.nstr(cmin, 12),
                c_mas=mp.nstr(cmax, 12), kappa_bar=mp.nstr(kbar, 12))


def limite_real(ell, n_ini=20000, semilla=0):
    """Mayor kappa para el que W decrece en Xi_ell: inf P/R por muestreo y Nelder--Mead.

    Es una cota superior del umbral que admite esta W (no del umbral dinámico real).
    """
    import numpy as np
    from scipy.optimize import minimize
    Xf, Yf, Af, Bf, zf = (float(v) for v in (X0, Y0, A, B0, zeta))
    cf, qf, af, ef, thf, phf, hf, Zf = (float(v) for v in (c, q, a, e, th, ph, h, Z))

    def obj(v):
        lx, ly, z = v
        if not 0 <= z < zf:
            return 1e9
        x, y = np.exp(lx), np.exp(ly)
        xh_, yh_ = Xf + Af * z, Yf - Bf * z
        W = cf * (x - xh_ - xh_ * np.log(x / xh_)) + af * (y - yh_ - yh_ * np.log(y / yh_))             - (z - 1) - (zf - 1) * np.log((zf - z) / (zf - 1))
        s = thf * x + phf * y
        P = cf * qf * (x - xh_) ** 2 + af * ef * (y - yh_) ** 2
        R = (Zf * s / (hf + s) - z) * (-cf * Af * np.log(x / xh_) + af * Bf * np.log(y / yh_)
                                        + (z - 1) / (zf - z))
        return P / R if (W <= ell and R > 0) else 1e9
    rng = np.random.default_rng(semilla)
    mejor, arg = 1e9, None
    for _ in range(n_ini):
        v = [rng.uniform(-3, 1.5), rng.uniform(-18, 1.5), rng.uniform(0, zf)]
        if obj(v) < 1e8:
            r = minimize(obj, v, method="Nelder-Mead",
                         options=dict(xatol=1e-10, fatol=1e-14, maxiter=2000))
            if r.fun < mejor:
                mejor, arg = r.fun, r.x
    return dict(ell=ell, inf_P_sobre_R=float(mejor),
                punto=[float(np.exp(arg[0])), float(np.exp(arg[1])), float(arg[2])])


def main():
    res = dict(ejemplo="racional (13)", z_estrella=mp.nstr(zs, 5), zeta=mp.nstr(zeta, 15),
               Gprima=mp.nstr(Gp, 15),
               niveles=[cotas(l) for l in ("0.05", "0.5", "2")],
               kappa_menos=0.4072941099458642,
               limite_de_esta_W=[limite_real(l) for l in (0.05, 0.5)],
               alcance=("Cotas explícitas con 50 dígitos y ensanchamiento hacia afuera; la "
                        "subdivisión da cotas rigurosas porque en cada subintervalo se usan "
                        "monotonías de (zeta - z) y de Phi."))
    (RAIZ / "Resultados" / "limites-globales" / "umbral-lento.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
