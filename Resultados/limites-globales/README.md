# Límites singulares: control computacional de los teoremas 13 y 15

Fecha: 18/09/2026. Modalidad C; afirmación CN-5 (apoya AD-14 y AD-15).
Esta carpeta contiene cálculos realizados, no una demostración.

## Reproducción

Desde la raíz del proyecto:

```powershell
python Codigo/limites_globales.py            # 288 órbitas, figura y resumen (~31 min)
python Codigo/limites_globales.py --solo-figura
python Codigo/tabla_limites.py               # informe/tabla-limites.tex y MBE_template/tabla-limits-en.tex
```

Entorno: Python 3.14.4, NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.8.
Semilla 20260918. El hash SHA-256 del script está en `resumen.json`.

## Protocolo

- Ejemplos: E1 = ejemplo racional (11); E2 = forma adimensional
  (mu,gamma,epsilon,chi,Zcal,vartheta,psi)=(99/20,100,1/100,1/10,3/4,10,2),
  equilibrio (1/20,1/2,9/20), ventana (0.24569, 4.40719), z#/zeta = 0.790.
- Velocidades: f*kappa_- con f en {0.02,0.1,0.3,0.6,0.9} y f*kappa_+ con
  f en {1.1,2,10,100}; 16 datos iniciales aleatorios por velocidad.
- Datos: ln x0 y ln y0 uniformes entre ln 1e-3 y ln(1.2 K), ln(1.2 y_K); z0 en [0, 1.2 Z].
- Radau en (ln x, ln y, z), rtol 1e-10, atol 1e-12; parada al entrar en la bola
  de radio 1e-4 alrededor de E*; horizonte 40/kappa + 60 + 45/|rho|.
- `umbral_lyapunov_malla`: inf P/R en malla 160^3 del subnivel W<=l con W=V+Psi.
  Solo el nivel 0.5 queda en el interior de la malla (x,y>=1e-4); los niveles
  mayores tocan el borde de la malla y no deben interpretarse.

## Resultados observados

- 288 órbitas, 0 sin converger (`convergencia-global.csv`).
- Mínimos de ln y hasta -3851 (E1, kappa=0.00815, paso retardado con z0>zeta)
  y hasta -1569 (E2, datos con y0 del orden de y_K ~ 9.5e3).
- Umbral de la función compuesta en W<=0.5: 2.1e-4 (E1) y 6.4e-4 (E2), frente a
  kappa_- = 0.407 y 0.246. La cota del teorema 13 es conservadora.

## Límites

No prueba estabilidad global ni excluye atractores fuera de los datos muestreados.
Los datos con z0 >= zeta de E1 convergen, pero el teorema 13 no los cubre.

## Ampliación del 19/09/2026 (revisión del artículo)

- `Codigo/limites_globales.py` (vigente para el artículo):
  - Paraleliza las órbitas, con 64 datos por velocidad y 1152 órbitas en total.
  - Añade métricas de cuasi-extinción: fracción de órbitas con ln y < -10 y tiempo máximo t_q.
  - Salidas: `convergencia-global-64.csv` y `resumen-64.json`. Todas las órbitas convergen.
  - Ejemplo I: t_q ~ 11/kappa en el régimen lento.
  - Ejemplo II: t_q ~ 350 a toda velocidad; esa cuasi-extinción es intrínseca a la dinámica (x, y), no la produce la regulación.
- `Codigo/limites_globales_16.py`:
  - Versión de 16 datos que reproduce `convergencia-global.csv` y `resumen.json`, en los que se basa la tabla del informe.
- `Codigo/umbral_lento.py` → `umbral-lento.json`: umbral lento garantizado, con cotas explícitas y redondeo hacia afuera.
  - kappa_slow >= 2.06e-3 en Xi_0.05 (0.854 <= z <= 1.080).
  - La cota degenera a 8e-16 en Xi_0.5 y a 2.4e-85 en Xi_2, porque el subnivel se acerca a z = zeta.
- `Codigo/tabla_limites.py`:
  - Por defecto genera solo la tabla del artículo, a partir de las salidas -64.
  - Con `--informe` genera la tabla del informe en su formato original.
