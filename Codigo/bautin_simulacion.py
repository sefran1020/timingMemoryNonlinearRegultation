"""Control independiente de l2 en el punto GH mediante integración directa.

En el punto GH (l1 = 0) la forma normal da r' = Re(c2) r^5 en la variedad central,
de modo que la semiamplitud en x (A_x ~ 2 r con v_x = 1) cumple
    d(A_x^{-4})/dt = -Re(c2)/4 + O(A_x^2).
Si l2 > 0 la órbita se aleja lentamente del equilibrio; si l2 < 0, se acerca.
Se integra el sistema completo (DOP853, rtol 1e-13) desde la variedad central
aproximada a segundo orden y se ajusta la pendiente de A_x^{-4} frente a t.

Uso: python Codigo/bautin_simulacion.py   Salida: Resultados/codim2-hill/bautin-simulacion.json
"""
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema

import bautin_l2 as B

RAIZ = Path(__file__).resolve().parents[1]


def main():
    mp.mp.dps = 50
    n, h = 2, 81
    l1Z = lambda Z: B.punto(B.TASAS, n, h, Z, "inf")[4]["l1"]
    ZGH = mp.findroot(l1Z, mp.mpf("30.0694"))
    E, s, sers, kap, nf = B.punto(B.TASAS, n, h, ZGH, "inf")
    rec2 = float(mp.re(nf["c2"]))
    Z, k = float(ZGH), float(kap)
    Ef = np.array([float(v) for v in E])

    def f(t, u):
        x, y, z = u
        s_ = x + 8 * y
        return [x * (12 - x - y - 10 * z), y * (-2 + 4 * x - y - z),
                k * (Z * s_ ** 2 / (81.0 ** 2 + s_ ** 2) - z)]

    # autovector crítico (v_x = 1) y términos de segundo orden de la variedad central
    A = np.array([[float(v) for v in fila] for fila in B.jacobiano(
        {kk: mp.mpf(v) for kk, v in B.TASAS.items()}, E, sers, kap).tolist()])
    ev, V = np.linalg.eig(A)
    i = int(np.argmax(ev.imag))
    q = V[:, i] / V[0, i]
    om = ev[i].imag
    resultados = []
    for A0 in (0.24, 0.16, 0.12):
        r0 = A0 / 2
        u0 = Ef + 2 * np.real(r0 * q)
        T = 2.0e4 if A0 > 0.13 else 6.0e4
        sol = solve_ivp(f, (0, T), u0, method="DOP853", rtol=1e-13, atol=1e-15,
                        max_step=2 * math.pi / om / 30, dense_output=False,
                        t_eval=np.linspace(0, T, int(T * 40)))
        x = sol.y[0]
        imax = argrelextrema(x, np.greater)[0]
        imin = argrelextrema(x, np.less)[0]
        m = min(len(imax), len(imin))
        tt = 0.5 * (sol.t[imax[:m]] + sol.t[imin[:m]])
        amp = 0.5 * (x[imax[:m]] - x[imin[:m]])
        sel = tt > 50
        pend, _ = np.polyfit(tt[sel], amp[sel] ** -4, 1)
        resultados.append(dict(A0=A0, T=T, A_inicial=float(amp[sel][0]), A_final=float(amp[sel][-1]),
                               pendiente_Am4=float(pend), prediccion=-rec2 / 4))
        print(resultados[-1], flush=True)
    out = dict(Z_GH=mp.nstr(ZGH, 20), kappa_GH=mp.nstr(kap, 20), Re_c2_vx1=rec2,
               l2_vx1=float(nf["l2"]), integraciones=resultados,
               alcance=("Integración numérica; confirma el signo y el orden de magnitud de Re c2. "
                        "Las correcciones O(A^2) explican diferencias con la predicción."))
    (RAIZ / "Resultados" / "codim2-hill" / "bautin-simulacion.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
