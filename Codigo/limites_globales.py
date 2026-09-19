"""Controles computacionales de los teoremas de estabilidad global (AD-13 a AD-17).

Modalidad C. Este programa NO demuestra los teoremas 13--16 del informe; produce
(i) las constantes exactas de los dos ejemplos racionales, (ii) un experimento de
convergencia desde datos iniciales aleatorios fuera de la ventana de Hopf,
(iii) una evaluación numérica del umbral de velocidad asociado al funcional
compuesto W = V + Psi del teorema 16 y (iv) la figura del informe.

Uso, desde la raíz del proyecto:
    python Codigo/limites_globales.py            # todo
    python Codigo/limites_globales.py --rapido   # menos datos iniciales (control)

Salidas en Resultados/limites-globales/ y figures/limits-global.pdf. Semilla fija: 20260918.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "Resultados" / "limites-globales"
FIGURAS = RAIZ / "figures" / "es"
SEMILLA = 20260918
# Radio de llegada: dentro de él decide la estabilidad local exponencial (teorema 3).
RADIO = 1e-4

# Parámetros dimensionales (r, q, a, b, m, c, e, d, theta, phi, h, Z), q = r/K.
EJEMPLOS = {
    "E1": dict(r=Fr(12), q=Fr(1), a=Fr(1), b=Fr(10), m=Fr(2), c=Fr(4), e=Fr(1),
               d=Fr(1), theta=Fr(1), phi=Fr(8), h=Fr(81), Z=Fr(10)),
    # Ejemplo 2: forma adimensional con (u*,v*,w*)=(1/20,1/2,9/20).
    "E2": dict(r=Fr(1), q=Fr(1), a=Fr(1), b=Fr(1), m=Fr(99, 20), c=Fr(100),
               e=Fr(1, 100), d=Fr(1, 10), theta=Fr(10), phi=Fr(2), h=Fr(1),
               Z=Fr(3, 4)),
}
EQUILIBRIOS = {"E1": (Fr(1), Fr(1), Fr(1)), "E2": (Fr(1, 20), Fr(1, 2), Fr(9, 20))}


def constantes_exactas(p: dict, eq: tuple) -> dict:
    """Cantidades racionales exactas; verifica que eq sea equilibrio."""
    r, q, a, b, m, c, e, d = (p[k] for k in "r q a b m c e d".split())
    th, ph, h, Z = p["theta"], p["phi"], p["h"], p["Z"]
    X, Y, W = eq
    s = th * X + ph * Y
    assert r - q * X - a * Y - b * W == 0
    assert -m + c * X - e * Y - d * W == 0
    assert Z * s / (h + s) == W
    D = q * e + a * c
    gp = Z * h / (h + s) ** 2
    al, be = th * gp, ph * gp
    T = q * X + e * Y
    Q = X * Y * D
    B = T + b * X * al + d * Y * be
    C = X * Y * (D + (b * e - a * d) * al + (q * d + b * c) * be)
    L = T * B + Q - C
    disc = L * L - 4 * B * T * Q
    zeta = (c * r - q * m) / (c * b + q * d)
    K = r / q
    yK = (c * K - m) / e
    sM = th * K + ph * yK
    zsharp = Z * sM / (h + sM)
    Gp = (th * (a * d - b * e) - ph * (c * b + q * d)) / D
    X0, Y0 = (e * r + a * m) / D, (c * r - q * m) / D
    A, B0 = (a * d - b * e) / D, (c * b + q * d) / D
    F0 = Z * (th * X0 + ph * Y0) / (h + th * X0 + ph * Y0)
    return dict(T=T, Q=Q, B=B, C=C, L=L, disc=disc, zeta=zeta, K=K, yK=yK,
                zsharp=zsharp, Gprime=Gp, X0=X0, Y0=Y0, A=A, B0=B0, F0=F0,
                alpha=al, beta=be, N1=B - T, N0=C - Q,
                ventana=bool(L < 0 and disc > 0),
                capacidad_moderada=bool(zsharp < zeta))


def kappas(cte: dict) -> tuple[float, float]:
    B, L, disc = float(cte["B"]), float(cte["L"]), float(cte["disc"])
    return (-L - math.sqrt(disc)) / (2 * B), (-L + math.sqrt(disc)) / (2 * B)


def campo_log(p: dict, kap: float):
    """Campo en coordenadas (ln x, ln y, z): evita el subdesbordamiento de y."""
    r, q, a, b, m, c, e, d, th, ph, h, Z = (float(p[k]) for k in
        "r q a b m c e d theta phi h Z".split())

    def f(_t, u):
        # El recorte solo actúa en iteraciones de Newton fuera del dominio útil.
        x, y, z = math.exp(min(u[0], 50.0)), math.exp(min(u[1], 50.0)), u[2]
        s = th * x + ph * y
        return [r - q * x - a * y - b * z, -m + c * x - e * y - d * z,
                kap * (Z * s / (h + s) - z)]

    def jac(_t, u):
        x, y = math.exp(min(u[0], 50.0)), math.exp(min(u[1], 50.0))
        s = th * x + ph * y
        gp = Z * h / (h + s) ** 2
        return [[-q * x, -a * y, -b], [c * x, -e * y, -d],
                [kap * gp * th * x, kap * gp * ph * y, -kap]]

    return f, jac


def abscisa(cte: dict, kap: float) -> float:
    T, Q, B, C = (float(cte[k]) for k in "T Q B C".split())
    return float(np.max(np.roots([1.0, T + kap, Q + B * kap, C * kap]).real))


UMBRAL_CUASI = -10.0   # ln y < -10 se registra como cuasi-extinción


def _orbita(args):
    """Trabajador paralelo: integra una órbita y devuelve sus métricas."""
    p, kap, u0, eq, horizonte = args
    Xs, Ys, Ws = eq
    f, jac = campo_log(p, kap)

    def cerca(_t, u):
        return math.dist((math.exp(u[0]), math.exp(u[1]), u[2]), (Xs, Ys, Ws)) - RADIO
    cerca.terminal = True
    sol = solve_ivp(f, (0.0, horizonte), u0, method="Radau", jac=jac,
                    rtol=1e-10, atol=1e-12, events=cerca, dense_output=True)
    uf = sol.y[:, -1]
    dist = math.dist((math.exp(uf[0]), math.exp(uf[1]), uf[2]), (Xs, Ys, Ws))
    tt = np.linspace(0.0, sol.t[-1], 20001)
    lny = sol.sol(tt)[1]
    t_cuasi = float(np.sum(lny < UMBRAL_CUASI) * (tt[1] - tt[0]))
    return dict(dist=dist, lny_min=float(sol.y[1].min()), z_max=float(sol.y[2].max()),
                t_final=float(sol.t[-1]), fallo=int(dist > 1.001 * RADIO or sol.status < 0),
                t_cuasi=t_cuasi)


def experimento(nombre: str, p: dict, eq, cte, lista_kappa, n_datos: int, rng, procesos=16):
    """Integra n_datos órbitas por velocidad (en paralelo) y registra métricas por velocidad."""
    from concurrent.futures import ProcessPoolExecutor
    filas = []
    eqf = tuple(float(v) for v in eq)
    K, yK, Z = float(cte["K"]), float(cte["yK"]), float(p["Z"])
    pf = {k: float(v) for k, v in p.items()}
    for kap in lista_kappa:
        rho = abscisa(cte, kap)
        assert rho < 0, "velocidad dentro de la ventana"
        horizonte = 40.0 / kap + 60.0 + 45.0 / abs(rho)
        datos = [[math.log(10 ** rng.uniform(-3, math.log10(1.2 * K))),
                  math.log(10 ** rng.uniform(-3, math.log10(1.2 * yK))),
                  rng.uniform(0.0, 1.2 * Z)] for _ in range(n_datos)]
        with ProcessPoolExecutor(max_workers=procesos) as ex:
            res = list(ex.map(_orbita, [(pf, kap, u0, eqf, horizonte) for u0 in datos]))
        tc = np.array([r["t_cuasi"] for r in res])
        filas.append(dict(ejemplo=nombre, kappa=kap, abscisa_espectral=rho,
                          datos=n_datos, no_convergen=sum(r["fallo"] for r in res),
                          distancia_final_max=max(r["dist"] for r in res),
                          ln_y_min=min(r["lny_min"] for r in res),
                          z_max=max(r["z_max"] for r in res),
                          t_final_max=max(r["t_final"] for r in res), horizonte=horizonte,
                          frac_cuasi=float(np.mean(tc > 0)), t_cuasi_max=float(tc.max()),
                          t_cuasi_mediana=float(np.median(tc[tc > 0])) if np.any(tc > 0) else 0.0))
        fl = filas[-1]
        print(f"{nombre} kappa={kap:<8g} rho={rho:+.3e} fallos={fl['no_convergen']}/{n_datos} "
              f"ln_y_min={fl['ln_y_min']:.1f} frac_cuasi={fl['frac_cuasi']:.2f} "
              f"t_cuasi_max={fl['t_cuasi_max']:.0f}", flush=True)
    return filas


def umbral_lyapunov(p: dict, cte: dict, eq, niveles, n=160):
    """inf P/R sobre {W<=l}: mayor kappa para el que dW/dt<0 con W=V+Psi (mu=1).

    Evaluación en malla, no certificada. P = cq xi^2 + ae eta^2,
    R = (S-z) (dV/dz + Psi'(z)); dW/dt = -P + kappa R.
    """
    r, q, a, b, m, c, e, d, th, ph, h, Z = (float(p[k]) for k in
        "r q a b m c e d theta phi h Z".split())
    X0, Y0, A, B0, zeta = (float(cte[k]) for k in "X0 Y0 A B0 zeta".split())
    zs = float(eq[2])
    lx = np.linspace(math.log(1e-4), math.log(3 * float(cte["K"])), n)
    ly = np.linspace(math.log(1e-4), math.log(3 * float(cte["yK"])), n)
    zz = np.linspace(0.0, zeta, n + 1)[:-1]  # malla en [0, zeta)
    x, y, z = np.meshgrid(np.exp(lx), np.exp(ly), zz, indexing="ij")
    xh, yh = X0 + A * z, Y0 - B0 * z
    V = c * (x - xh - xh * np.log(x / xh)) + a * (y - yh - yh * np.log(y / yh))
    Psi = -(z - zs) - (zeta - zs) * np.log((zeta - z) / (zeta - zs))
    W = V + Psi
    s = th * x + ph * y
    Pq = c * q * (x - xh) ** 2 + a * e * (y - yh) ** 2
    R = (Z * s / (h + s) - z) * (-c * A * np.log(x / xh) + a * B0 * np.log(y / yh)
                                 + (z - zs) / (zeta - z))
    salida = []
    for nivel in niveles:
        mask = (W <= nivel) & (R > 0) & (Pq > 1e-10)
        salida.append(dict(nivel=nivel, puntos=int(mask.sum()),
                           kappa_malla=float(np.min(Pq[mask] / R[mask])) if mask.any() else None,
                           x_min=float(x[W <= nivel].min()), y_min=float(y[W <= nivel].min()),
                           z_max=float(z[W <= nivel].max())))
    return salida


TEXTOS = {
    "es": dict(a=r"(a) $\kappa=0.05$: paso retardado por $z=\zeta$", tl=r"tiempo lento $\kappa t$",
               b=r"(b) $z(t)$; línea discontinua: $z_*$; punteada: $\zeta$", red="reducido $z=S$",
               c="(c) ajuste rápido frente al sistema reducido",
               m0=r"$M_0$: $y=\hat y(z)$"),
    "en": dict(a=r"(a) $\kappa=0.05$: delayed passage through $z=\zeta$", tl=r"slow time $\kappa t$",
               b=r"(b) $z(t)$; dashed: $z_*$; dotted: $\zeta$", red="reduced $z=S$",
               c="(c) fast adjustment versus the reduced system",
               m0=r"$\Upsilon_0$: $y=\hat y(z)$"),
}


def figura(p1, cte1, p2, idioma="es"):
    tx = TEXTOS[idioma]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "legend.fontsize": 9})
    fig = plt.figure(figsize=(8.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05])
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])]
    zeta, X0, Y0, A, B0 = (float(cte1[k]) for k in "zeta X0 Y0 A B0".split())
    r, q, b = float(p1["r"]), float(p1["q"]), float(p1["b"])
    # (a) límite lento: paso retardado
    kap = 0.05
    f, jac = campo_log(p1, kap)
    for u0, col in (([math.log(6.0), math.log(20.0), 4.0], "C0"),
                    ([math.log(0.5), math.log(3.0), 0.2], "C1")):
        sol = solve_ivp(f, (0, 900), u0, method="Radau", jac=jac, rtol=1e-10,
                        atol=1e-12, dense_output=True)
        t = np.linspace(0, 900, 6000)
        u = sol.sol(t)
        ax[0].plot(u[2], u[1] / math.log(10), col, lw=1.1)
        ax[1].plot(kap * t, u[2], col, lw=1.1)
    zg = np.linspace(0, zeta, 200)
    ax[0].plot(zg[:-1], np.log10(Y0 - B0 * zg[:-1]), "k--", lw=1, label=tx["m0"])
    ax[0].axvline(zeta, color="gray", lw=0.8, ls=":")
    ax[0].set_xlabel("$z$"); ax[0].set_ylabel(r"$\log_{10}y$")
    ax[0].set_title(tx["a"], fontsize=9)
    ax[0].legend(fontsize=8, loc="lower left")
    ax[1].axhline(zeta, color="gray", lw=0.8, ls=":"); ax[1].axhline(1.0, color="k", lw=0.8, ls="--")
    ax[1].set_xlabel(tx["tl"]); ax[1].set_ylabel("$z$")
    ax[1].set_title(tx["b"], fontsize=9)
    # (c) límite rápido: colapso sobre el flujo reducido
    th, ph, h, Z = (float(p1[k]) for k in ("theta", "phi", "h", "Z"))
    a_, c_, e_, d_, m_ = (float(p1[k]) for k in "a c e d m".split())

    def red(_t, u):
        x, y = math.exp(u[0]), math.exp(u[1]); s = th * x + ph * y; S = Z * s / (h + s)
        return [r - q * x - a_ * y - b * S, -m_ + c_ * x - e_ * y - d_ * S]
    for kap, col, lab in ((500.0, "C0", r"$\kappa=500$"), (12.0, "C2", r"$\kappa=12$")):
        f, jac = campo_log(p1, kap)
        for i, u0 in enumerate(([math.log(8.0), math.log(0.05), 0.0], [math.log(0.2), math.log(12.0), 9.0])):
            sol = solve_ivp(f, (0, 40), u0, method="Radau", jac=jac, rtol=1e-10, atol=1e-12,
                            t_eval=np.linspace(0, 40, 4000))
            ax[2].plot(np.exp(sol.y[0]), np.exp(sol.y[1]), col, lw=1.0, label=lab if i == 0 else None)
    for i, u0 in enumerate(([math.log(8.0), math.log(0.05)], [math.log(0.2), math.log(12.0)])):
        sol = solve_ivp(red, (0, 40), u0, method="Radau", rtol=1e-10, atol=1e-12,
                        t_eval=np.linspace(0, 40, 4000))
        ax[2].plot(np.exp(sol.y[0]), np.exp(sol.y[1]), "k--", lw=0.9,
                   label=tx["red"] if i == 0 else None)
    ax[2].plot([1], [1], "ko", ms=4)
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xlabel("$x$"); ax[2].set_ylabel("$y$")
    ax[2].set_title(tx["c"], fontsize=9)
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    if idioma == "es":
        destinos = (FIGURAS / "limites-globales.pdf", SALIDA / "limites-globales.png")
    else:
        destinos = (RAIZ / "figures" / "limits-global.pdf",
                    SALIDA / "limits-global-en.png")
    for destino in destinos:
        fig.savefig(destino, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--datos", type=int, default=64)
    ap.add_argument("--solo-figura", action="store_true")
    args = ap.parse_args()
    if args.solo_figura:
        SALIDA.mkdir(parents=True, exist_ok=True)
        for idioma in ("es", "en"):
            figura(EJEMPLOS["E1"], constantes_exactas(EJEMPLOS["E1"], EQUILIBRIOS["E1"]),
                   EJEMPLOS["E2"], idioma)
        return
    SALIDA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMILLA)
    n_datos = 3 if args.rapido else args.datos
    t0 = time.time()
    resumen = {"semilla": SEMILLA, "n_datos_por_velocidad": n_datos, "ejemplos": {}}
    filas = []
    for nombre, p in EJEMPLOS.items():
        eq = EQUILIBRIOS[nombre]
        cte = constantes_exactas(p, eq)
        km, kp = kappas(cte)
        lentas = [km * f for f in (0.02, 0.1, 0.3, 0.6, 0.9)]
        rapidas = [kp * f for f in (1.1, 2.0, 10.0, 100.0)]
        resumen["ejemplos"][nombre] = {
            "parametros": {k: str(v) for k, v in p.items()},
            "equilibrio": [str(v) for v in eq],
            "exactas": {k: (str(v) if isinstance(v, Fr) else v) for k, v in cte.items()},
            "kappa_menos": km, "kappa_mas": kp,
            "umbral_lyapunov_malla": umbral_lyapunov(p, cte, eq, [0.5, 2.0, 8.0, 32.0]),
        }
        print(nombre, "kappa-,kappa+ =", km, kp, "zsharp/zeta =",
              float(cte["zsharp"] / cte["zeta"]), flush=True)
        filas += experimento(nombre, p, eq, cte, lentas + rapidas, n_datos, rng)
    with open(SALIDA / f"convergencia-global-{n_datos}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader(); w.writerows(filas)
    for idioma in ("es", "en"):
        figura(EJEMPLOS["E1"], constantes_exactas(EJEMPLOS["E1"], EQUILIBRIOS["E1"]),
               EJEMPLOS["E2"], idioma)
    resumen["total_orbitas"] = sum(f["datos"] for f in filas)
    resumen["total_no_convergen"] = sum(f["no_convergen"] for f in filas)
    resumen["entorno"] = {"python": platform.python_version(), "numpy": np.__version__,
                          "scipy": scipy.__version__}
    resumen["sha256_script"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    resumen["segundos"] = round(time.time() - t0, 1)
    resumen["alcance"] = ("Observación numérica con integrador Radau en coordenadas logarítmicas; "
                          "no demuestra estabilidad global ni excluye atractores no muestreados.")
    (SALIDA / f"resumen-{n_datos}.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    print("total no convergen:", resumen["total_no_convergen"], "de", resumen["total_orbitas"])


if __name__ == "__main__":
    sys.exit(main())
