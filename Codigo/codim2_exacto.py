"""Certificados exactos de la línea de codimensión dos (AD-18 a AD-22).

Modalidad T/C. Álgebra exacta con SymPy; los aislamientos numéricos se indican
como tales. Produce Resultados/codim2-hill/certificados.json.

1. Identidades de la dicotomía del lazo: C = -XYD F'(z*), C - Q = -XYD Z g' G'.
2. Retardo de Erlang de orden k: P_k(0) = (k kappa)^k C,
   P_k'(0) = k (k kappa)^(k-1) (Q + B kappa); límite lento (condición secante).
3. Familia de Hill ajustada al equilibrio del ejemplo: sigma_2(n), sigma_3(n)
   y signo exacto de 2 omega l1 para todo n > 9/10 en ambos Hopf.
4. Punto de Hopf degenerado (l1 = 0) en el plano (kappa, sigma_3) con sigma_2 = -1/450.
5. Ventanas de Erlang k = 1..8 para el ejemplo racional (raíces aisladas numéricamente).

Uso: python Codigo/codim2_exacto.py
"""
from __future__ import annotations

import json
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "Resultados" / "codim2-hill"

# Coeficientes exactos de 2 omega l1 = a0 + a1 s2 + a2 s2^2 + a3 s3 en kappa_{+-}
# (Resultados/extension-hopf/lyapunov-clase.json, AD-8).
K = sp.Symbol("kappa")
A_EXPR = [sp.sympify(e) for e in (
    "-5*(758633276142334*kappa + 30735199500588101)/275119308100591677",
    "-(3097896890641965962728*kappa - 22060621134815780572825)/106471172234928978999",
    "12850*(11810811800269872194*kappa - 101897365381749926045)/106471172234928978999",
    "-2570*(1686630076*kappa - 15572443045)/50213333853",
)]
KM = (261 - sp.sqrt(52921)) / 76
KP = (261 + sp.sqrt(52921)) / 76


def signo_exacto(expr) -> int:
    """Signo exacto de p + q*sqrt(52921), p, q racionales."""
    raiz = sp.sqrt(52921)
    e = sp.expand(sp.radsimp(expr))
    q = e.coeff(raiz)
    pr = sp.expand(e - q * raiz)
    assert pr.is_Rational and q.is_Rational, e
    sp_, sq = sp.sign(pr), sp.sign(q)
    if sq == 0 or sp_ == sq:
        return int(sp_ if sp_ != 0 else sq)
    if sp_ == 0:
        return int(sq)
    # signos opuestos: decide la comparación de cuadrados
    return int(sp_) if pr**2 > 52921 * q**2 else int(sq)


def determinante_erlang(k: int) -> str:
    """Compara P_k con det(lambda I - J) del sistema (k+2)-dimensional."""
    q, a, b, c, e, d, X, Y, al, be, kap, lam = sp.symbols(
        "q a b c e d X Y alpha beta kappa lambda", positive=True)
    n = k + 2
    J = sp.zeros(n, n)
    J[0, 0], J[0, 1], J[0, n - 1] = -q * X, -a * X, -b * X
    J[1, 0], J[1, 1], J[1, n - 1] = c * Y, -e * Y, -d * Y
    J[2, 0], J[2, 1], J[2, 2] = k * kap * al, k * kap * be, -k * kap
    for j in range(3, n):
        J[j, j - 1], J[j, j] = k * kap, -k * kap
    det = (lam * sp.eye(n) - J).det()
    D = q * e + a * c
    T, Q = q * X + e * Y, X * Y * D
    B = T + b * X * al + d * Y * be
    C = X * Y * (D + (b * e - a * d) * al + (q * d + b * c) * be)
    P = (lam**2 + T * lam + Q) * (lam + k * kap)**k + (k * kap)**k * ((B - T) * lam + (C - Q))
    return str(sp.expand(det - P))


def identidades_lazo():
    q, a, b, c, e, d, th, ph, X, Y, Z, gp = sp.symbols(
        "q a b c e d theta phi X Y Z gprime", positive=True)
    D = q * e + a * c
    al, be = Z * th * gp, Z * ph * gp
    T = q * X + e * Y
    Q = X * Y * D
    B = T + b * X * al + d * Y * be
    C = X * Y * (D + (b * e - a * d) * al + (q * d + b * c) * be)
    Gp = (th * (a * d - b * e) - ph * (c * b + q * d)) / D
    Fp = Z * gp * Gp - 1
    lam, kap = sp.symbols("lambda kappa", positive=True)
    res = {"C_mas_XYD_Fprima": str(sp.simplify(C + X * Y * D * Fp)),
           "CmenosQ_mas_XYDZgG": str(sp.simplify(C - Q + X * Y * D * Z * gp * Gp)),
           "N1_igual_BmenosT": str(sp.simplify((B - T) - (b * X * al + d * Y * be)))}
    erl = {}
    for k in range(1, 7):
        P = (lam**2 + T * lam + Q) * (lam + k * kap)**k + (k * kap)**k * ((B - T) * lam + (C - Q))
        erl[k] = {"P0_menos": str(sp.simplify(P.subs(lam, 0) - (k * kap)**k * C)),
                  "dP0_cociente": str(sp.simplify(sp.diff(P, lam).subs(lam, 0)
                                                  / (k * (k * kap)**(k - 1) * (Q + B * kap))))}
    res["erlang"] = erl
    res["erlang_determinante_menos_Pk"] = {k: determinante_erlang(k) for k in range(1, 5)}
    assert all(v == "0" for v in res["erlang_determinante_menos_Pk"].values())
    # Con C = 0 (pliegue) y k = 1: P = lambda (lambda^2 + (T+kappa) lambda + Q + B kappa).
    P1 = lam**3 + (T + kap) * lam**2 + (Q + B * kap) * lam
    res["pliegue_k1_factor"] = str(sp.factor(P1))
    return res


def familia_hill():
    n = sp.Symbol("n", positive=True)
    U, s = sp.symbols("U s", positive=True)
    d1 = n * U * (1 - U) / s

    def D(expr):
        return sp.diff(expr, s) + sp.diff(expr, U) * d1
    d2, d3 = sp.simplify(D(d1)), None
    d3 = sp.simplify(D(d2))
    Zc = 1 / (1 - sp.Rational(9, 10) / n)
    sub = {U: 1 / Zc, s: 9}
    s1 = sp.simplify((Zc * d1).subs(sub))
    s2 = sp.factor(sp.simplify((Zc * d2).subs(sub)))
    s3 = sp.factor(sp.simplify((Zc * d3).subs(sub)))
    assert sp.simplify(s1 - sp.Rational(1, 10)) == 0
    out = {"sigma1": str(s1), "sigma2": str(s2), "sigma3": str(s3),
           "Z(n)": str(Zc), "h^n": "9^n (Z-1)"}
    ramas = {}
    for nombre, kv in (("inferior", KM), ("superior", KP)):
        a0, a1, a2, a3 = (sp.nsimplify(sp.expand(ae.subs(K, kv))) for ae in A_EXPR)
        poly = sp.expand(a0 + a1 * s2 + a2 * s2**2 + a3 * s3)
        pc = sp.Poly(poly, n)
        c2, c1, c0 = (sp.radsimp(sp.expand(v)) for v in pc.all_coeffs())
        # Certificado en n >= 9/10: c2 < 0, vértice <= 9/10 y P(9/10) < 0; entonces
        # P es decreciente en [9/10, oo) y negativo. Signos decididos exactamente
        # en Q(sqrt(52921)) con signo_exacto (comparación de cuadrados).
        c2f, c1f, c0f = (float(sp.N(v, 50)) for v in (c2, c1, c0))
        vert = sp.radsimp(-c1 / (2 * c2))
        p910 = sp.radsimp(poly.subs(n, sp.Rational(9, 10)))
        ok_c2 = signo_exacto(c2) < 0
        # Con c2 < 0: vértice -c1/(2 c2) <= 9/10  <=>  -c1 >= (9/5) c2.
        ok_vert = signo_exacto(-c1 - sp.Rational(9, 5) * c2) >= 0
        ok_p = signo_exacto(p910) < 0
        assert ok_c2 == bool(sp.N(c2, 60) < 0) and ok_p == bool(sp.N(p910, 60) < 0)
        assert ok_vert == bool(sp.N(vert - sp.Rational(9, 10), 60) <= 0)
        raices = [complex(r) for r in sp.Poly(sp.N(poly, 40), n).nroots()]
        ramas[nombre] = {"coef_n2": str(c2), "coef_n1": str(c1), "coef_n0": str(c0),
                         "coef_float": [c2f, c1f, c0f],
                         "vertice_n": float(sp.N(vert, 30)),
                         "P_9_10": float(sp.N(p910, 30)),
                         "raices_aprox": [[r.real, r.imag] for r in raices],
                         "c2_negativo": ok_c2, "vertice_menor_9_10": ok_vert,
                         "P_9_10_negativo": ok_p,
                         "valor_n1": float(sp.N(poly.subs(n, 1), 30))}
        assert ok_c2 and ok_vert and ok_p, nombre
    out["ramas"] = ramas
    return out


def hopf_degenerado_jet():
    s2 = sp.Rational(-1, 450)
    res = {}
    for nombre, kv in (("inferior", KM), ("superior", KP)):
        a0, a1, a2, a3 = (sp.radsimp(sp.expand(ae.subs(K, kv))) for ae in A_EXPR)
        s3 = sp.radsimp(-(a0 + a1 * s2 + a2 * s2**2) / a3)
        res[nombre] = {"sigma3_estrella": str(sp.nsimplify(s3)),
                       "sigma3_estrella_float": float(sp.N(s3, 40)),
                       "a3_float": float(sp.N(a3, 30)),
                       "dl1_dsigma3_signo": "positivo" if signo_exacto(a3) > 0 else "negativo"}
    return res


def ventanas_erlang():
    T, Q, B, C = 2.0, 5.0, 3.8, 38.7
    N1, N0 = B - T, C - Q
    filas = []
    for k in range(1, 9):
        def P(lam, kap):
            return (lam**2 + T * lam + Q) * (lam + k * kap)**k + (k * kap)**k * (N1 * lam + N0)

        def abscisa(kap):
            p = np.polynomial.polynomial
            coef = p.polymul([Q, T, 1.0], p.polypow([k * kap, 1.0], k))
            coef = p.polyadd(coef, (k * kap)**k * np.array([N0, N1]))
            return float(np.max(np.roots(coef[::-1]).real))
        ks = np.logspace(-5, 3, 20001)
        v = np.array([abscisa(x) for x in ks])
        cambios = np.where(np.sign(v[1:]) != np.sign(v[:-1]))[0]
        from scipy.optimize import brentq
        raices = [brentq(abscisa, ks[i], ks[i + 1], xtol=1e-14) for i in cambios]
        umbral = None if k < 3 else float(1 / np.cos(np.pi / k)**k)
        filas.append({"k": k, "N0_sobre_Q": N0 / Q, "sec_k": umbral,
                      "inestable_lento_predicho": bool(umbral is not None and N0 / Q > umbral),
                      "abscisa_en_1e-5": float(v[0]), "cruces": raices})
    # órdenes altos: solo el extremo superior (el inferior es 0 porque sec^k(pi/k) < N0/Q)
    from scipy.optimize import brentq
    p = np.polynomial.polynomial
    for k in (16, 32, 64):
        def abscisa(kap, k=k):
            coef = p.polyadd(p.polymul([Q, T, 1.0], p.polypow([k * kap, 1.0], k)),
                             (k * kap)**k * np.array([N0, N1]))
            return float(np.max(np.roots(coef[::-1]).real))
        sup = brentq(abscisa, 8.5, 12.0, xtol=1e-12)
        filas.append({"k": k, "N0_sobre_Q": N0 / Q, "sec_k": float(1 / np.cos(np.pi / k)**k),
                      "inestable_lento_predicho": True, "abscisa_en_1e-5": None,
                      "cruces": [sup], "nota": "solo extremo superior"})
    return filas


def retardo_discreto():
    """Límite k -> infinito (retardo discreto tau = 1/kappa) en el ejemplo racional.

    Delta_tau(lambda) = lambda^2 + T lambda + Q + e^{-lambda tau}(N1 lambda + N0).
    f(omega) = |Q - omega^2 + i T omega|^2 - |N0 + i N1 omega|^2; los cruces imaginarios
    ocurren en las raíces positivas de f y su dirección es sign f'(omega_c).
    """
    T, Q, N1, N0 = sp.Integer(2), sp.Integer(5), sp.Rational(9, 5), sp.Rational(337, 10)
    w = sp.Symbol("omega", positive=True)
    f = sp.expand((Q - w**2)**2 + T**2 * w**2 - N0**2 - N1**2 * w**2)
    raices = [rt for rt in sp.solve(sp.Eq(f, 0), w) if rt.is_positive]
    assert len(raices) == 1
    wc = sp.nsimplify(raices[0])
    wc2 = sp.simplify(wc**2)
    dfp = sp.diff(f, w).subs(w, wc)
    wn = mp.mpf(str(sp.N(wc, 40)))
    Dv = mp.mpc(float(Q) - wn**2, 2 * wn)
    Nv = mp.mpc(mp.mpf(337) / 10, mp.mpf(9) / 5 * wn)
    fase = mp.arg(-Dv / Nv)                  # e^{-i w tau} = -D/N
    tau0 = (-fase) / wn
    while tau0 <= 0:
        tau0 += 2 * mp.pi / wn
    return dict(f=str(f), omega_c2=str(wc2), omega_c=float(sp.N(wc, 20)),
                fprima_signo=int(sp.sign(sp.N(dfp, 30))), tau_c=mp.nstr(tau0, 15),
                kappa_c=mp.nstr(1 / tau0, 15), periodo_cruces=mp.nstr(2 * mp.pi / wn, 15),
                N0_sobre_Q=float(N0 / Q))


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    datos = {"sympy": sp.__version__, "numpy": np.__version__,
             "identidades": identidades_lazo(), "hill": familia_hill(),
             "hopf_degenerado_jet": hopf_degenerado_jet(),
             "erlang_ejemplo": ventanas_erlang(),
             "retardo_discreto": retardo_discreto(),
             "alcance": ("Las identidades y el signo de la familia de Hill son exactos; "
                         "las ventanas de Erlang k>=2 son raíces aisladas numéricamente.")}
    (SALIDA / "certificados.json").write_text(json.dumps(datos, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    print(json.dumps({k: datos[k] for k in ("hill", "hopf_degenerado_jet")}, indent=1)[:3000])
    for f in datos["erlang_ejemplo"]:
        print(f["k"], f["sec_k"], f["inestable_lento_predicho"], f["abscisa_en_1e-5"], f["cruces"])


if __name__ == "__main__":
    main()
