"""Histéresis por barrido cuasiestático de kappa y robustez del pliegue de ciclos grandes.

Objetivo de Hill n=2, h=81, con las tasas de Example I (Codigo/bautin_l2.py, TASAS).

1. Barrido cuasiestático en Z=45: kappa se reduce y luego se aumenta en pasos de 0.001.
   Cada paso integra T=3000 unidades de tiempo desde el estado final del paso anterior.
   En el barrido ascendente se añade un empujón de 1e-4 en x, para no depender de la
   lentitud del crecimiento cerca de kappa_- (paso lento por el Hopf). La semiamplitud de x
   se mide en las últimas 300 unidades.
2. Robustez: en Z=45 se perturban b y phi en +-5 %. kappa_- se obtiene con la fórmula exacta
   (bautin_l2.punto), junto con el signo de l1 en kappa_-. El pliegue de ciclos grandes se
   acota por el mismo barrido descendente: kappa_LPC queda entre el último kappa con
   oscilación grande (semiamplitud > 0.5) y el primero sin ella.

La cota del barrido se compara con el kappa_LPC de la continuación (lpc-resumen.csv) en el
caso base.

Uso: python Codigo/histeresis_robustez.py
Salidas: Resultados/codim2-hill/histeresis-robustez.json,
         Resultados/codim2-hill/barrido-kappa-Z45.csv,
         figures/sweep.pdf
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "Codigo"))
import bautin_l2 as B  # noqa: E402

DAT = RAIZ / "Resultados" / "codim2-hill"
Z = 45.0
T, TMED, PASO = 3000.0, 300.0, 0.001


def campo(p, k):
    r, q, a, b, m, c, e, d, th, ph = (float(p[x]) for x in "r q a b m c e d th ph".split())

    def f(t, u):
        x, y, z = u
        s = th * x + ph * y
        return [x * (r - q * x - a * y - b * z), y * (-m + c * x - e * y - d * z),
                k * (Z * s * s / (6561 + s * s) - z)]
    return f


def paso(p, k, u):
    f = campo(p, k)
    sol = solve_ivp(f, (0, T), u, method="LSODA", rtol=1e-10, atol=1e-13, dense_output=True)
    xs = sol.sol(np.linspace(T - TMED, T, 6000))[0]
    return sol.y[:, -1], float((xs.max() - xs.min()) / 2)


def kappa_menos(p):
    mp.mp.dps = 30
    E, s, sers, kap, nf = B.punto(p, 2, 81, mp.mpf(Z), "inf")
    return [float(v) for v in E], float(kap), float(nf["l1"])


def barrido_abajo(p, k0, kfin, u0):
    """Desde k0 (dentro de la ventana) hacia abajo; devuelve filas (kappa, semiamplitud)."""
    filas, u, k = [], np.array(u0, float), k0
    while k >= kfin - 1e-12:
        u, A = paso(p, k, u)
        filas.append((round(k, 6), A))
        k -= PASO
    return filas


def barrido_arriba(p, k0, kfin, u0):
    filas, u, k = [], np.array(u0, float), k0
    while k <= kfin + 1e-12:
        u = u + np.array([1e-4, 0, 0])
        u, A = paso(p, k, u)
        filas.append((round(k, 6), A))
        k += PASO
    return filas


def cota_lpc(filas, umbral=0.5):
    grandes = [k for k, A in filas if A > umbral]
    pequenas = [k for k, A in filas if A <= umbral]
    k_ult = min(grandes)
    k_pri = max(k for k in pequenas if k < k_ult)
    return k_pri, k_ult


def main():
    base = dict(B.TASAS)
    casos = {"base": base,
             "b*0.95": dict(base, b=base["b"] * 0.95), "b*1.05": dict(base, b=base["b"] * 1.05),
             "phi*0.95": dict(base, ph=base["ph"] * 0.95), "phi*1.05": dict(base, ph=base["ph"] * 1.05)}
    salida = {"Z": Z, "T": T, "paso": PASO, "casos": {}}
    for nombre, p in casos.items():
        E, km, l1 = kappa_menos(p)
        k0 = round(km + 0.02, 3)
        # arranque cerca del equilibrio inestable dentro de la ventana: el atractor es el ciclo grande
        u0 = [E[0] * 1.1, E[1], E[2]]
        filas = barrido_abajo(p, k0, round(km - 0.05, 3), u0)
        lo, hi = cota_lpc(filas)
        salida["casos"][nombre] = dict(E=E, kappa_menos=km, l1_en_kappa_menos=l1,
                                       kappa_LPC_intervalo=[lo, hi],
                                       brecha_min=km - hi, biestable=bool(hi < km),
                                       filas=filas)
        print(nombre, "kappa_-=%.5f l1=%.3g LPC in [%.3f, %.3f]" % (km, l1, lo, hi), flush=True)
        if nombre == "base":
            abajo = filas
            arriba = barrido_arriba(p, filas[-1][0], k0, [E[0], E[1], E[2]])
    salida["casos"]["base"]["kappa_LPC_continuacion"] = 0.2731462280176249
    with open(DAT / "barrido-kappa-Z45.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sentido", "kappa", "semiamplitud_x"])
        for k, A in abajo:
            w.writerow(["abajo", k, A])
        for k, A in arriba:
            w.writerow(["arriba", k, A])
    salida["barrido_arriba_Z45"] = arriba
    (DAT / "histeresis-robustez.json").write_text(json.dumps(salida, indent=1), encoding="utf-8")

    km = salida["casos"]["base"]["kappa_menos"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot([k for k, _ in abajo], [A for _, A in abajo], "v-", color="C3", ms=3.5, lw=1,
            label=r"$\kappa$ decreasing")
    ax.plot([k for k, _ in arriba], [A for _, A in arriba], "^-", color="C0", ms=3.5, lw=1,
            label=r"$\kappa$ increasing")
    ax.axvline(km, color="0.4", lw=0.7, ls=":")
    ax.axvline(0.2731462280176249, color="0.4", lw=0.7, ls=":")
    ymax = max(A for _, A in abajo)
    ax.text(km, 0.45 * ymax, r"$\kappa_-$ ", ha="right", fontsize=10)
    ax.text(0.2731462280176249, 0.55 * ymax, r"$\kappa_{\rm LPC}$ ", ha="right", fontsize=10)
    ax.set_xlabel(r"speed $\kappa$")
    ax.set_ylabel(r"semi-amplitude of $x$")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(RAIZ / "figures" / "sweep.pdf")
    fig.savefig(DAT / "barrido-kappa-Z45.png", dpi=150)


if __name__ == "__main__":
    main()
