"""Realización exacta de la ruptura de la dicotomía del lazo con retardo de Erlang k = 2.

Modalidad T/C (aritmética racional exacta con SymPy). Se construyen parámetros
positivos de M2_k y un objetivo de Hill con n = 2 tales que:

  (a) E* = (1, 1/2, 15) es un equilibrio interior (r, m > 0);
  (b) C = 0 (pliegue: lazo positivo, G' > 0);
  (c) con k = 2 y kappa = 1, el polinomio característico factoriza como
      lambda (lambda + rho) (lambda^2 + omega^2): cero simple junto con un par imaginario;
      es decir, un punto espectral pliegue--Hopf, imposible con k = 1 (teorema del artículo);
  (d) con k = 1 (memoria exponencial) y el mismo equilibrio, no hay par imaginario;
  (e) perturbando g'(s*) a 99/100 de su valor: C > 0, lazo positivo, y con k = 2 existe
      kappa_H > 0 con par imaginario simple que cruza transversalmente (Hopf con lazo positivo).

Uso: python Codigo/erlang_realizacion.py   Salida: Resultados/codim2-hill/erlang-realizacion.json
"""
import json
from fractions import Fraction as Fr
from pathlib import Path

import sympy as sp

RAIZ = Path(__file__).resolve().parents[1]
lam, kap = sp.symbols("lambda kappa")


def coeficientes(p, X, Y, g1):
    q, a, b, c, e, d, th, ph = (p[k] for k in "q a b c e d th ph".split())
    D = q * e + a * c
    al, be = th * g1, ph * g1
    T = q * X + e * Y
    Q = X * Y * D
    B = T + b * X * al + d * Y * be
    C = X * Y * (D + (b * e - a * d) * al + (q * d + b * c) * be)
    Gp = (th * (a * d - b * e) - ph * (c * b + q * d)) / D
    return T, Q, B, C, Gp


def Pk(T, Q, B, C, k, kv=None):
    kk = kap if kv is None else kv
    return sp.expand((lam**2 + T * lam + Q) * (lam + k * kk)**k + (k * kk)**k * ((B - T) * lam + (C - Q)))


def main():
    X, Y, W = Fr(1), Fr(1, 2), Fr(15)
    p = dict(q=Fr(1, 5), a=Fr(4), c=Fr(4), e=Fr(1, 10), b=Fr(1, 4), d=Fr(1, 5))
    kappa0 = Fr(1)
    # alfa, beta que resuelven C = 0 y la condición de Hopf del cúbico con k = 2 (g'(s*) = 1)
    al, be = Fr(16417, 750), Fr(2179, 2400)
    p["th"], p["ph"] = al, be
    q, a, b, c, e, d = (p[k] for k in "q a b c e d".split())
    r = q * X + a * Y + b * W
    m = c * X - e * Y - d * W
    s_star = p["th"] * X + p["ph"] * Y
    g1 = Fr(1)
    # objetivo de Hill n = 2 con g(s*) = W y g'(s*) = 1:  g = Zc s^2/(H + s^2)
    n = 2
    u = 1 - g1 * s_star / (n * W)
    Zc = W / u
    H = s_star**2 * (1 - u) / u
    s = sp.Symbol("s")
    g = sp.Rational(Zc.numerator, Zc.denominator) * s**2 / (sp.Rational(H.numerator, H.denominator) + s**2)
    ss = sp.Rational(s_star.numerator, s_star.denominator)
    chk_g = (sp.simplify(g.subs(s, ss) - W), sp.simplify(sp.diff(g, s).subs(s, ss) - 1))
    # (a) equilibrio
    assert r > 0 and m > 0 and 0 < u < 1
    eq1 = r - q * X - a * Y - b * W
    eq2 = -m + c * X - e * Y - d * W
    assert eq1 == 0 and eq2 == 0 and chk_g == (0, 0)
    # (b)
    T, Q, B, C, Gp = coeficientes(p, X, Y, g1)
    assert C == 0 and Gp > 0
    # (c) k = 2, kappa = 1
    P2 = sp.factor(Pk(T, Q, B, C, 2, kappa0))
    cubico = sp.Poly(sp.cancel(Pk(T, Q, B, C, 2, kappa0) / lam), lam)
    a1, a2, a3 = [cubico.all_coeffs()[i] for i in (1, 2, 3)]
    hopf_cubico = sp.simplify(a1 * a2 - a3)
    omega2 = sp.nsimplify(a3 / a1)
    assert hopf_cubico == 0 and a1 > 0 and a2 > 0 and a3 > 0
    # (d) k = 1: P1 = lambda (lambda^2 + (T+kappa) lambda + Q + B kappa), sin par imaginario
    P1 = sp.factor(Pk(T, Q, B, C, 1))
    # (e) perturbación: g'(s*) = 99/100
    g1e = Fr(99, 100)
    Te, Qe, Be, Ce, Gpe = coeficientes(p, X, Y, g1e)
    assert Ce > 0 and Gpe > 0
    b1, b2, b3, b4 = sp.Poly(Pk(Te, Qe, Be, Ce, 2), lam).all_coeffs()[1:]
    cond = sp.factor(sp.expand(b3 * (b1 * b2 - b3) - b1**2 * b4))   # borde de Hurwitz del cuártico
    raices = [rt for rt in sp.Poly(sp.numer(sp.together(cond)), kap).real_roots() if rt > 0]
    raices_f = [float(rt.evalf(30)) for rt in raices]
    cruces = []
    for rt in raices:
        kv = rt.evalf(40)
        ev = sp.Poly(Pk(Te, Qe, Be, Ce, 2, kv), lam).nroots(n=30)
        im = [z for z in ev if abs(sp.re(z)) < 1e-20]
        # transversalidad: derivada de la parte real del par crítico respecto de kappa
        h = sp.Float("1e-12", 40)
        def maxre(kx):
            return max(sp.re(z) for z in sp.Poly(Pk(Te, Qe, Be, Ce, 2, kx), lam).nroots(n=30)
                       if abs(sp.im(z)) > 1e-6)
        dre = (maxre(kv + h) - maxre(kv - h)) / (2 * h)
        cruces.append(dict(kappa=float(kv), autovalores=[str(sp.N(z, 12)) for z in ev],
                           par_imaginario=[str(sp.N(z, 12)) for z in im],
                           dRe_dkappa=float(dre)))
    out = dict(
        parametros={k: str(v) for k, v in dict(p, r=r, m=m).items()},
        equilibrio=[str(X), str(Y), str(W)], s_star=str(s_star),
        objetivo_hill=dict(n=n, Z=str(Zc), H_igual_h_n=str(H), g_s_star=str(W), gprima_s_star="1"),
        coeficientes=dict(T=str(T), Q=str(Q), B=str(B), C=str(C), Gprima=str(Gp)),
        k2_kappa1=dict(factorizacion=str(P2), a1=str(a1), a2=str(a2), a3=str(a3),
                       a1a2_menos_a3=str(hopf_cubico), omega2=str(omega2),
                       omega=float(sp.sqrt(omega2))),
        k1_factorizacion=str(P1),
        perturbacion=dict(gprima=str(g1e), C=str(Ce), Gprima=str(Gpe),
                          kappas_borde_hurwitz=raices_f, cruces=cruces),
        alcance=("Aritmética racional exacta para (a)-(d); en (e) las raíces de la condición "
                 "de borde se aíslan exactamente (real_roots) y los autovalores se evalúan a 30 dígitos."))
    (RAIZ / "Resultados" / "codim2-hill" / "erlang-realizacion.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
