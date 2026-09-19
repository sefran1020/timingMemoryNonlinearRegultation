"""Segundo coeficiente de Lyapunov l2 en el punto de Hopf generalizado (GH).

Modalidad C (cálculo de alta precisión, no intervalar). Resuelve las ecuaciones
homológicas de la variedad central y de la forma normal de Poincaré hasta orden
cinco, sin derivadas numéricas:

    u(w, wb) = q w + qb wb + sum_{2<=j+k<=5} U_jk w^j wb^k,
    w' = i*omega*w + c1 w^2 wb + c2 w^3 wb^2,

con q autovector derecho, p autovector adjunto y conj(p)^T q = 1. Entonces
l1 = Re(c1)/omega y l2 = Re(c2)/omega (con la convención de Kuznetsov,
G21 = 2 c1 y G32 = 12 c2). Cuando l1 = 0, el signo de l2 no depende de la
normalización. El campo de M2 es polinomial salvo el objetivo g; para la
respuesta de Hill g(s) = Z s^n/(h^n+s^n) con n entero, la serie de Taylor de g
se obtiene por división exacta de series. No hay diferencias finitas.

Validación interna: en el ejemplo racional (n = 1, h = 81, Z = 10), l1 debe
coincidir con los valores certificados -0.1953474016 y -0.0632669257 (v_x = 1).

Uso: python Codigo/bautin_l2.py      Salida: Resultados/codim2-hill/bautin-l2.json
"""
from __future__ import annotations

import json
from pathlib import Path

import mpmath as mp

mp.mp.dps = 60
RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "Resultados" / "codim2-hill"
ORDEN = 5
TASAS = dict(r=12, q=1, a=1, b=10, m=2, c=4, e=1, d=1, th=1, ph=8)


# ---------- series en una variable (Taylor del objetivo) ----------
def serie_hill(sstar, n, h, Z, orden=ORDEN):
    """Coeficientes g_j/j! de g(s*+t) = Z (s*+t)^n / (h^n + (s*+t)^n), j=0..orden."""
    num = [mp.binomial(n, j) * mp.mpf(sstar) ** (n - j) if j <= n else mp.mpf(0)
           for j in range(orden + 1)]
    den = [num[j] + (mp.mpf(h) ** n if j == 0 else 0) for j in range(orden + 1)]
    out = []
    for j in range(orden + 1):
        acc = num[j] - sum(out[i] * den[j - i] for i in range(j))
        out.append(acc / den[0])
    return [Z * v for v in out]


# ---------- polinomios en (w, wb) con coeficientes vectoriales o escalares ----------
def pmul(A, B):
    C = {}
    for (a1, b1), x in A.items():
        for (a2, b2), y in B.items():
            k = (a1 + a2, b1 + b2)
            if k[0] + k[1] > ORDEN:
                continue
            C[k] = C.get(k, 0) + x * y
    return C


def padd(*Ps):
    C = {}
    for P in Ps:
        for k, v in P.items():
            C[k] = C.get(k, 0) + v
    return C


def pscale(P, s):
    return {k: s * v for k, v in P.items()}


def componente(U, i):
    return {k: v[i] for k, v in U.items()}


def campo_serie(U, p, E, sers, kap):
    """F(E+u) truncado; U: dict (j,k) -> mp.matrix(3,1). Devuelve dict de 3-vectores."""
    X, Y, W = E
    u1, u2, u3 = (componente(U, i) for i in range(3))
    q, a, b, c, e, d, th, ph = (p[k] for k in "q a b c e d th ph".split())
    lin1 = padd(pscale(u1, -q), pscale(u2, -a), pscale(u3, -b))
    lin2 = padd(pscale(u1, c), pscale(u2, -e), pscale(u3, -d))
    F1 = padd(pscale(lin1, X), pmul(u1, lin1))
    F2 = padd(pscale(lin2, Y), pmul(u2, lin2))
    ell = padd(pscale(u1, th), pscale(u2, ph))
    F3 = pscale(u3, -1)
    pot = {(0, 0): mp.mpf(1)}
    for j in range(1, ORDEN + 1):
        pot = pmul(pot, ell)
        F3 = padd(F3, pscale(pot, sers[j]))
    F3 = pscale(F3, kap)
    claves = set(F1) | set(F2) | set(F3)
    return {k: mp.matrix([F1.get(k, 0), F2.get(k, 0), F3.get(k, 0)]) for k in claves}


def jacobiano(p, E, sers, kap):
    X, Y, W = E
    q, a, b, c, e, d, th, ph = (p[k] for k in "q a b c e d th ph".split())
    g1 = sers[1]
    return mp.matrix([[-q * X, -a * X, -b * X], [c * Y, -e * Y, -d * Y],
                      [kap * g1 * th, kap * g1 * ph, -kap]])


def forma_normal(p, E, sers, kap, normaliza_vx=True):
    A = jacobiano(p, E, sers, kap)
    ev, V = mp.eig(A)
    i = max(range(3), key=lambda j: mp.im(ev[j]))
    om = mp.im(ev[i])
    assert abs(mp.re(ev[i])) < mp.mpf(10) ** (-(mp.mp.dps - 20)), "no es un punto de Hopf"
    qv = V[:, i]
    qv = qv / (qv[0] if normaliza_vx else mp.norm(qv))
    evl, Wl = mp.eig(A.T)
    j = min(range(3), key=lambda t: abs(evl[t] + 1j * om))
    pv = Wl[:, j]
    s = sum(mp.conj(pv[t]) * qv[t] for t in range(3))
    pv = pv / mp.conj(s)                      # conj(p)^T q = 1
    dotp = lambda z: sum(mp.conj(pv[t]) * z[t] for t in range(3))
    I3 = mp.eye(3)
    U = {(1, 0): qv, (0, 1): mp.matrix([mp.conj(x) for x in qv])}
    cs = {}                                   # c[(j,k)] del término resonante
    for n in range(2, ORDEN + 1):
        N = campo_serie(U, p, E, sers, kap)
        for jj in range(n, -1, -1):
            kk = n - jj
            rhs = N.get((jj, kk), mp.matrix(3, 1)) - A * U.get((jj, kk), mp.matrix(3, 1)) * 0
            # términos no lineales de la forma normal aplicados a U de orden menor
            for (a_, b_), Uab in list(U.items()):
                for (r_, cr) in cs.items():           # w' tiene c_r w^{r} wb^{r-1}
                    # contribución de a*U_ab*c_r*w^{a-1+r} wb^{b+r-1}
                    kmon = (a_ - 1 + r_, b_ + r_ - 1)
                    if kmon == (jj, kk) and a_ >= 1:
                        rhs = rhs - a_ * cr * Uab
                    kmon2 = (a_ + r_ - 1, b_ - 1 + r_)
                    if kmon2 == (jj, kk) and b_ >= 1:
                        rhs = rhs - b_ * mp.conj(cr) * Uab
            M = (1j * om * (jj - kk)) * I3 - A
            if jj - kk == 1:                         # resonante: w^{m} wb^{m-1}
                c_new = dotp(rhs)
                cs[jj] = c_new
                rhs = rhs - c_new * qv
                # sistema con borde: M U + s q = rhs, conj(p)^T U = 0
                Bm = mp.matrix(4, 4)
                for r0 in range(3):
                    for c0 in range(3):
                        Bm[r0, c0] = M[r0, c0]
                    Bm[r0, 3] = qv[r0]
                    Bm[3, r0] = mp.conj(pv[r0])
                sol = mp.lu_solve(Bm, mp.matrix([rhs[0], rhs[1], rhs[2], 0]))
                U[(jj, kk)] = mp.matrix([sol[0], sol[1], sol[2]])
            elif jj - kk == -1:
                U[(jj, kk)] = mp.matrix([mp.conj(x) for x in U[(kk, jj)]])
            else:
                U[(jj, kk)] = mp.lu_solve(M, rhs)
    c1, c2 = cs.get(2, mp.mpc(0)), cs.get(3, mp.mpc(0))
    return dict(omega=om, c1=c1, c2=c2, l1=mp.re(c1) / om, l2=mp.re(c2) / om,
                qnorm2=sum(abs(x) ** 2 for x in qv),
                residuo=residuo(U, om, c1, c2, p, E, sers, kap))


def residuo(U, om, c1, c2, p, E, sers, kap):
    """Control independiente: max |F(u) - u_w w' - u_wb wb'| en grados 1..5."""
    wdot = {(1, 0): 1j * om, (2, 1): c1, (3, 2): c2}
    wbdot = {(0, 1): -1j * om, (1, 2): mp.conj(c1), (2, 3): mp.conj(c2)}
    du_w = {(j - 1, k): j * v for (j, k), v in U.items() if j >= 1}
    du_wb = {(j, k - 1): k * v for (j, k), v in U.items() if k >= 1}
    lhs = padd(pmul(du_w, wdot), pmul(du_wb, wbdot))
    rhs = campo_serie(U, p, E, sers, kap)
    peor = mp.mpf(0)
    for key in set(lhs) | set(rhs):
        if sum(key) > ORDEN:
            continue
        dif = rhs.get(key, mp.matrix(3, 1)) - lhs.get(key, mp.matrix(3, 1))
        peor = max(peor, max(abs(x) for x in dif))
    return peor


# ---------- equilibrio y ventana para la respuesta de Hill ----------
def equilibrio(p, n, h, Z):
    r, q, a, b, m, c, e, d, th, ph = (mp.mpf(p[k]) for k in "r q a b m c e d th ph".split())
    D = q * e + a * c
    X0, Y0, A, B0 = (e * r + a * m) / D, (c * r - q * m) / D, (a * d - b * e) / D, (c * b + q * d) / D
    zeta = (c * r - q * m) / (c * b + q * d)
    G = lambda z: th * (X0 + A * z) + ph * (Y0 - B0 * z)
    F = lambda z: Z * G(z) ** n / (h ** n + G(z) ** n) - z
    z = mp.findroot(F, (mp.mpf(0), zeta), solver="anderson")
    return (X0 + A * z, Y0 - B0 * z, z)


def coeficientes(p, E, g1):
    X, Y, W = E
    q, a, b, c, e, d, th, ph = (mp.mpf(p[k]) for k in "q a b c e d th ph".split())
    D = q * e + a * c
    al, be = th * g1, ph * g1
    T = q * X + e * Y
    Q = X * Y * D
    B = T + b * X * al + d * Y * be
    C = X * Y * (D + (b * e - a * d) * al + (q * d + b * c) * be)
    L = T * B + Q - C
    disc = L * L - 4 * B * T * Q
    return T, Q, B, C, L, disc


def punto(p, n, h, Z, rama):
    E = equilibrio(p, n, h, Z)
    s = mp.mpf(p["th"]) * E[0] + mp.mpf(p["ph"]) * E[1]
    sers = serie_hill(s, n, h, Z)
    T, Q, B, C, L, disc = coeficientes(p, E, sers[1])
    kap = (-L + (1 if rama == "sup" else -1) * mp.sqrt(disc)) / (2 * B)
    pm = {k: mp.mpf(v) for k, v in p.items()}
    return E, s, sers, kap, forma_normal(pm, E, sers, kap)


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    res = {"mp_dps": mp.mp.dps, "orden": ORDEN}
    # 1) validación con el ejemplo racional
    val = {}
    for rama in ("inf", "sup"):
        E, s, sers, kap, nf = punto(TASAS, 1, 81, 10, rama)
        val[rama] = dict(kappa=mp.nstr(kap, 20), l1_vx1=mp.nstr(nf["l1"], 15),
                         l1_q_unitario=mp.nstr(nf["l1"] / nf["qnorm2"], 15))
    res["validacion_racional"] = val
    print("validación:", val)
    # 2) localización de GH en la rama inferior de Hill n=2, h=81
    n, h = 2, 81
    l1Z = lambda Z: punto(TASAS, n, h, Z, "inf")[4]["l1"]
    ZGH = mp.findroot(l1Z, mp.mpf("30.0694"), tol=mp.mpf(10) ** (-45))
    E, s, sers, kap, nf = punto(TASAS, n, h, ZGH, "inf")
    res["GH"] = dict(Z=mp.nstr(ZGH, 25), kappa=mp.nstr(kap, 25), omega=mp.nstr(nf["omega"], 20),
                     equilibrio=[mp.nstr(v, 20) for v in E], s_star=mp.nstr(s, 20),
                     l1=mp.nstr(nf["l1"], 5), l2_vx1=mp.nstr(nf["l2"], 20),
                     residuo_grado5=mp.nstr(nf["residuo"], 5),
                     Re_c2=mp.nstr(mp.re(nf["c2"]), 20))
    # 3) robustez: dependencia de la precisión y de la normalización
    checks = {}
    for dps in (40, 80):
        with mp.workdps(dps):
            Z2 = mp.findroot(l1Z, ZGH)
            nf2 = punto(TASAS, n, h, Z2, "inf")[4]
            checks[f"dps{dps}"] = dict(Z=mp.nstr(Z2, 20), l2=mp.nstr(nf2["l2"], 15))
    pm = {k: mp.mpf(v) for k, v in TASAS.items()}
    nfu = forma_normal(pm, E, sers, kap, normaliza_vx=False)
    checks["q_unitario"] = dict(l1=mp.nstr(nfu["l1"], 5), l2=mp.nstr(nfu["l2"], 15))
    res["controles"] = checks
    # 4) signo de l1 a ambos lados de Z_GH
    res["l1_lados"] = {z: mp.nstr(l1Z(mp.mpf(z)), 10) for z in ("29", "31")}
    res["alcance"] = ("Cálculo en precisión múltiple sin diferencias finitas; no es "
                      "aritmética de intervalos. El signo de l2 se considera establecido "
                      "si es estable ante precisión (40/60/80 dígitos) y normalización.")
    (SALIDA / "bautin-l2.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
