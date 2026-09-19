"""Genera las tablas de convergencia del informe (español) y del artículo (inglés).

Salida: tables/tabla-limits.tex (tabla I del artículo), a partir de
Resultados/limites-globales/. Uso: python Codigo/tabla_limites.py
"""
import csv
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "Resultados" / "limites-globales"

TEXTOS = {
    "es": dict(destino=RAIZ / "tables" / "es" / "tabla-limites.tex", label="inf:tabla-limites",
               caption=(r"Convergencia observada fuera de la ventana. $\rho$ es la abscisa "
                        r"espectral en $E_*$; se indican las órbitas que no alcanzaron la bola de "
                        r"radio $10^{-4}$, el mínimo de $\ln y$ y el mayor tiempo de llegada."),
               cab=r"Ejemplo & Régimen & $\kappa$ & $\rho$ & No convergen & $\min\ln y$ & $t_{\max}$\\",
               lento="lento", rapido="rápido", pos="H"),
    "en": dict(destino=RAIZ / "tables" / "tabla-limits.tex", label="tab:limits",
               caption=(r"Convergence and quasi-extinction outside the window (64 random initial "
                        r"data per speed). $\rho$ is the spectral abscissa at $E_*$; "
                        r"``not conv.'' counts orbits that did not reach the ball of radius $10^{-4}$; "
                        r"``share'' is the fraction of orbits with $\ln y<-10$ at some time and $t_q$ "
                        r"the longest time spent there."),
               cab=r"Ex. & Regime & $\kappa$ & $\rho$ & Not conv. & $\min\ln y$ & Share & $t_q$\\",
               lento="slow", rapido="fast", pos="t"),
}


def main():
    """Por defecto solo la tabla del artículo (64 datos por velocidad).

    Con --informe se regenera la tabla del informe desde la corrida de 16 datos.
    """
    import sys
    informe = "--informe" in sys.argv
    sufijo = "" if informe else "-64"
    filas = list(csv.DictReader(open(DATOS / f"convergencia-global{sufijo}.csv", encoding="utf-8")))
    resumen = json.loads((DATOS / f"resumen{sufijo}.json").read_text(encoding="utf-8"))
    for clave, tx in TEXTOS.items():
        if (clave == "es") != informe:
            continue
        lineas = [rf"\begin{{table}}[{tx['pos']}]", r"\centering\small",
                  rf"\caption{{{tx['caption']}}}", rf"\label{{{tx['label']}}}",
                  r"\begin{tabular}{llrrrrrr}", r"\toprule", tx["cab"], r"\midrule"]
        for f in filas:
            km = resumen["ejemplos"][f["ejemplo"]]["kappa_menos"]
            k = float(f["kappa"])
            regimen = tx["lento"] if k < km else tx["rapido"]
            if informe:   # formato original del informe (corrida de 16 datos)
                lineas.append(f"{f['ejemplo']} & {regimen} & {k:.4g} & "
                              f"{float(f['abscisa_espectral']):.3g} & "
                              f"{f['no_convergen']}/{f['datos']} & {float(f['ln_y_min']):.0f} & "
                              f"{float(f['t_final_max']):.0f}\\\\")
                continue
            ej = "I" if f["ejemplo"] == "E1" else "II"
            tq = float(f["t_cuasi_max"])
            lineas.append(f"{ej} & {regimen} & {k:.4g} & "
                          f"{float(f['abscisa_espectral']):.3g} & "
                          f"{f['no_convergen']}/{f['datos']} & {float(f['ln_y_min']):.0f} & "
                          f"{float(f['frac_cuasi']):.2f} & {tq:.1f}\\\\")
        lineas += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        tx["destino"].write_text("\n".join(lineas) + "\n", encoding="utf-8")
        print(tx["destino"])


if __name__ == "__main__":
    main()
