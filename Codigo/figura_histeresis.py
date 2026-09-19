"""Figura de histéresis inducida por la velocidad (objetivo de Hill n=2, h=81).

Lee Resultados/codim2-hill/lpc-resumen.csv y ciclos-Z*.csv (Codigo/lpc_hill_bk.jl),
Resultados/codim2-hill/bautin-l2.json (Codigo/bautin_l2.py) y recalcula la curva de
Hopf inferior kappa_-(Z) y el signo de l1 con Codigo/bautin_l2.py.

Salidas: figures/hysteresis.pdf (figura 3 del artículo), figures/variants/hysteresis-plain.pdf,
figures/es/histeresis.pdf,
Resultados/codim2-hill/histeresis.png, Resultados/codim2-hill/histeresis-datos.csv.
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
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "Codigo"))
import bautin_l2 as B  # noqa: E402

DAT = RAIZ / "Resultados" / "codim2-hill"
TXT = {
    "en": dict(hopf_sup="Hopf, supercritical", hopf_sub="Hopf, subcritical",
               lpc_g="fold of large cycles", lpc_l="fold of small cycles", gh="GH ($l_2>0$)",
               bi="bistable", xl="capacity $Z$", yl=r"speed $\kappa$",
               eq_s="stable equilibrium", eq_u="unstable equilibrium", cyc_s="stable cycle",
               cyc_u="unstable cycle", amp=r"semi-amplitude of $x$",
               ta=r"(a) lower window edge in the $(Z,\kappa)$ plane",
               tb=r"(c) $Z=45$: hysteresis in $\kappa$",
               tc=r"(b) offsets from the Hopf curve", dk=r"$\kappa-\kappa_-(Z)$"),
    "es": dict(hopf_sup="Hopf supercrítica", hopf_sub="Hopf subcrítica",
               lpc_g="pliegue de ciclos grandes", lpc_l="pliegue de ciclos pequeños",
               gh="GH ($l_2>0$)", bi="biestable", xl="capacidad $Z$", yl=r"velocidad $\kappa$",
               eq_s="equilibrio estable", eq_u="equilibrio inestable", cyc_s="ciclo estable",
               cyc_u="ciclo inestable", amp=r"semiamplitud de $x$",
               ta=r"(a) borde inferior de la ventana en $(Z,\kappa)$",
               tb=r"(c) $Z=45$: histéresis en $\kappa$",
               tc=r"(b) desplazamiento respecto de Hopf", dk=r"$\kappa-\kappa_-(Z)$"),
}
# Variante para Chaos (AIP): rótulos explícitos de LPC y GH.
TXT["aip"] = dict(TXT["en"], lpc_g=r"LPC: fold of large cycles, $\kappa_{\rm LPC}(Z)$",
                  lpc_l="fold of small cycles", gh=r"GH: Bautin point ($l_2>0$)",
                  hopf_sup=r"Hopf $\kappa_-(Z)$, supercritical",
                  hopf_sub=r"Hopf $\kappa_-(Z)$, subcritical")


def curva_hopf(Zs):
    mp.mp.dps = 30
    out = []
    for Z in Zs:
        E, s, sers, kap, nf = B.punto(B.TASAS, 2, 81, mp.mpf(Z), "inf")
        out.append((Z, float(kap), float(nf["l1"])))
    return np.array(out)


def pliegues():
    res = pd.read_csv(DAT / "lpc-resumen.csv").drop_duplicates("Z").sort_values("Z")
    # Un giro de la rama solo es pliegue de ciclos grandes si la amplitud en el giro es
    # apreciable; para Z <= 28.5 el único giro es el rebote de la rama en el Hopf.
    validos = []
    for Z in res.Z:
        d = pd.read_csv(DAT / f"ciclos-Z{Z:.2f}.csv")
        k = d.kappa.values
        g0 = np.where(np.sign(np.diff(k[1:])) != np.sign(np.diff(k[:-1])))[0] + 1
        validos.append(len(g0) > 0 and (d.xmax.values[g0[0]] - d.xmin.values[g0[0]]) / 2 > 0.1)
    res = res[np.array(validos)]
    loc = []
    for Z in res.Z:
        f = DAT / f"ciclos-Z{Z:.2f}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        k = d.kappa.values
        giros = np.where(np.sign(np.diff(k[1:])) != np.sign(np.diff(k[:-1])))[0] + 1
        if len(giros) >= 2 and d.stable.values[giros[1]] and abs(k[giros[1]] - k[giros[0]]) > 0:
            # segundo giro con ciclo estable: pliegue local de ciclos pequeños
            amp = (d.xmax.values[giros[1]] - d.xmin.values[giros[1]]) / 2
            if amp > 1e-3 and k[giros[1]] > k[giros[0]]:
                loc.append((Z, k[giros[1]], amp))
    return res, np.array(loc)


def figura(idioma, hopf, res, loc, gh):
    t = TXT[idioma]
    plt.rcParams.update({"font.size": 11, "legend.fontsize": 9})
    fig = plt.figure(figsize=(8.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05])
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])]
    Z, km, l1 = hopf[:, 0], hopf[:, 1], hopf[:, 2]
    sup, sub = l1 < 0, l1 >= 0
    ax[0].plot(Z[sup], km[sup], "k-", lw=1.6, label=t["hopf_sup"])
    ax[0].plot(Z[sub], km[sub], "k--", lw=1.6, label=t["hopf_sub"])
    ax[0].plot(res.Z, res.kappa_LPC, "o-", color="C3", ms=3.5, lw=1.2, label=t["lpc_g"])
    if len(loc):
        ax[0].plot(loc[:, 0], loc[:, 1], "s-", color="C0", ms=3.5, lw=1.0, label=t["lpc_l"])
    kmi = np.interp(res.Z, Z, km)
    ax[0].fill_between(res.Z, res.kappa_LPC, np.maximum(kmi, res.kappa_LPC), color="C3",
                       alpha=0.15, label=t["bi"])
    ax[0].plot([gh["Z"]], [gh["kappa"]], "*", color="C1", ms=12, label=t["gh"], zorder=5)
    ax[0].set_xlabel(t["xl"]); ax[0].set_ylabel(t["yl"]); ax[0].set_title(t["ta"], fontsize=10)
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].set_xlim(Z.min(), Z.max())
    # panel b: desplazamientos respecto de la curva de Hopf, cerca del GH
    zc = (Z >= 28.3) & (Z <= 40.0)
    ax[1].plot(Z[zc & sup], 0 * Z[zc & sup], "k-", lw=2.2, zorder=4)
    ax[1].plot(Z[zc & sub], 0 * Z[zc & sub], "k--", lw=1.6, zorder=4)
    rr = res[(res.Z >= 28.3) & (res.Z <= 40.0)]
    ax[1].plot(rr.Z, rr.kappa_LPC - np.interp(rr.Z, Z, km), "o-", color="C3", ms=3.5, lw=1.2)
    if len(loc):
        ax[1].plot(loc[:, 0], loc[:, 1] - np.interp(loc[:, 0], Z, km), "s-", color="C0",
                   ms=3.5, lw=1.0)
    ax[1].plot([gh["Z"]], [0], "*", color="C1", ms=12, zorder=5)
    ax[1].set_xlabel(t["xl"]); ax[1].set_ylabel(t["dk"]); ax[1].set_title(t["tc"], fontsize=10)
    ax[1].set_xlim(28.3, 40.0)
    ax[1].set_yscale("symlog", linthresh=1e-4)
    ax[1].axvline(gh["Z"], color="C1", lw=0.6, ls=":")
    # panel c: diagrama de amplitud en Z = 45
    d = pd.read_csv(DAT / "ciclos-Z45.00.csv")
    k = d.kappa.values
    giros = np.where(np.sign(np.diff(k[1:])) != np.sign(np.diff(k[:-1])))[0] + 1
    fin = giros[1] + 1 if len(giros) > 1 else len(k)
    d = d.iloc[:fin]
    amp = (d.xmax - d.xmin) / 2
    # Estabilidad: rama grande estable hasta el primer pliegue e inestable después
    # (verificado con multiplicadores de Floquet por integración variacional
    # independiente; la clasificación de BifurcationKit falla cuando x(t) ~ 1e-5).
    st = np.arange(len(d)) < giros[0]
    for mask, sty, lab in ((st, "-", t["cyc_s"]), (~st, "--", t["cyc_u"])):
        kk = np.where(mask, d.kappa, np.nan)
        ax[2].plot(kk, np.where(mask, amp, np.nan), sty, color="C0", lw=1.5, label=lab)
    kmz = float(np.interp(45, Z, km))
    ax[2].plot([0.2, kmz], [0, 0], "k-", lw=2, label=t["eq_s"])
    ax[2].plot([kmz, 0.45], [0, 0], "k:", lw=2, label=t["eq_u"])
    klpc = float(res.loc[res.Z == 45, "kappa_LPC"].iloc[0])
    ax[2].axvspan(klpc, kmz, color="C3", alpha=0.15)
    if idioma == "aip":
        ytop = float(np.nanmax(amp))
        for kv, lab, ha in ((klpc, r"$\kappa_{\rm LPC}$ ", "right"), (kmz, r" $\kappa_-$", "left")):
            ax[2].axvline(kv, color="0.4", lw=0.6, ls=":")
            ax[2].text(kv, 0.9 * ytop, lab, ha=ha, va="bottom", fontsize=10)
        ax[1].text(gh["Z"], 1e-4, " GH", color="C1", fontsize=9, va="bottom")
    ax[2].set_xlim(0.2, 0.45)
    ax[2].set_xlabel(t["yl"]); ax[2].set_ylabel(t["amp"]); ax[2].set_title(t["tb"], fontsize=10)
    ax[2].legend(fontsize=8, loc="center right")
    fig.tight_layout()
    return fig


def main():
    gh_json = json.loads((DAT / "bautin-l2.json").read_text(encoding="utf-8"))["GH"]
    gh = dict(Z=float(gh_json["Z"]), kappa=float(gh_json["kappa"]))
    Zs = np.concatenate([np.linspace(26.0, 30.0, 17), np.linspace(30.1, 100.0, 60)])
    hopf = curva_hopf(Zs)
    res, loc = pliegues()
    with open(DAT / "histeresis-datos.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tipo", "Z", "kappa", "extra"])
        for Z, k, l in hopf:
            w.writerow(["hopf_inferior", Z, k, l])
        for _, row in res.iterrows():
            w.writerow(["pliegue_global", row.Z, row.kappa_LPC, row.kappa_menos])
        for Z, k, a in loc:
            w.writerow(["pliegue_local", Z, k, a])
    for idioma, destinos in (("en", [RAIZ / "figures" / "variants" / "hysteresis-plain.pdf"]),
                             ("es", [RAIZ / "figures" / "es" / "histeresis.pdf",
                                     DAT / "histeresis.png"]),
                             ("aip", [RAIZ / "figures" / "hysteresis.pdf"])):
        fig = figura(idioma, hopf, res, loc, gh)
        for dst in destinos:
            fig.savefig(dst, dpi=200)
        plt.close(fig)
    print("pliegues locales:", loc.tolist())


if __name__ == "__main__":
    main()
