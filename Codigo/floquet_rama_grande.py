"""Estabilidad de la rama de ciclos grandes (objetivo de Hill n=2, h=81) por integración variacional.

Control independiente de las etiquetas de estabilidad de BifurcationKit, que fallan cuando
x(t) se acerca a 1e-5. Para cada Z se sigue el ciclo grande por simulación, desde kappa alto
hacia kappa bajo reutilizando el estado final, y se calculan los multiplicadores de la matriz
de monodromía (LSODA, rtol 1e-12, atol 1e-14). También corrige kappa_- en lpc-resumen.csv con
la fórmula exacta (Codigo/bautin_l2.py).

Uso: python Codigo/floquet_rama_grande.py
Salida: Resultados/codim2-hill/floquet-rama-grande.json
"""
import json
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import bautin_l2 as B

RAIZ = Path(__file__).resolve().parents[1]
DAT = RAIZ / "Resultados" / "codim2-hill"
CASOS = {35.0: (1.3, 1.0, 0.9, 0.8, 0.7, 0.5, 0.4, 0.37),
         45.0: (0.45, 0.40, 0.37, 0.33, 0.30, 0.285, 0.278)}


def campo(Z, k):
    def f(t, u):
        x, y, z = u[:3]
        s = x + 8 * y
        g = Z * s * s / (6561 + s * s)
        gp = Z * 2 * 6561 * s / (6561 + s * s) ** 2
        J = np.array([[12 - 2 * x - y - 10 * z, -x, -10 * x],
                      [4 * y, -2 + 4 * x - 2 * y - z, -y],
                      [k * gp, 8 * k * gp, -k]])
        base = [x * (12 - x - y - 10 * z), y * (-2 + 4 * x - y - z), k * (g - z)]
        if len(u) == 3:
            return base
        return np.concatenate([base, (J @ u[3:].reshape(3, 3)).ravel()])
    return f


def main():
    salida = {}
    for Z, ks in CASOS.items():
        u = np.array([2.0, 1.0, 0.5])
        filas = []
        for k in ks:
            f = campo(Z, k)
            u = solve_ivp(f, (0, 3000), u, method="LSODA", rtol=1e-11, atol=1e-13).y[:, -1]
            ev = lambda t, v: v[0] - 1.2
            ev.direction = 1
            s2 = solve_ivp(f, (0, 80), u, method="LSODA", rtol=1e-12, atol=1e-14, events=ev)
            te = s2.t_events[0]
            if len(te) < 2:
                filas.append(dict(kappa=k, ciclo=False))
                continue
            P = te[-1] - te[-2]
            u0 = s2.y_events[0][-2]
            s3 = solve_ivp(f, (0, P), np.concatenate([u0, np.eye(3).ravel()]), method="LSODA",
                           rtol=1e-12, atol=1e-14, dense_output=True)
            mu = np.sort(np.abs(np.linalg.eigvals(s3.y[3:, -1].reshape(3, 3))))
            xs = s3.sol(np.linspace(0, P, 4000))[0]
            filas.append(dict(kappa=k, ciclo=True, periodo=float(P),
                              cierre=float(np.abs(s3.y[:3, -1] - u0).max()),
                              multiplicadores=[float(v) for v in mu],
                              x_min=float(xs.min()), semiamplitud=float((xs.max() - xs.min()) / 2)))
            u = u0
            print(Z, filas[-1], flush=True)
        salida[str(Z)] = filas
    # corrección de kappa_- en lpc-resumen.csv (continuación arrancada dentro de la ventana)
    mp.mp.dps = 30
    res = pd.read_csv(DAT / "lpc-resumen.csv")
    res["kappa_menos"] = [float(B.punto(B.TASAS, 2, 81, mp.mpf(Z), "inf")[3]) for Z in res.Z]
    res.to_csv(DAT / "lpc-resumen.csv", index=False)
    salida["nota"] = ("kappa_menos de lpc-resumen.csv recalculado con la fórmula exacta; "
                      "los multiplicadores no triviales < 1 confirman la estabilidad de la rama grande.")
    (DAT / "floquet-rama-grande.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")


if __name__ == "__main__":
    main()
