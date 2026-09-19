# Codimensión dos: certificados exactos y continuación con respuesta de Hill

Fecha: 18/09/2026. Modalidades T y C; afirmaciones AD-17 a AD-22, CN-4 y CN-6.

## Reproducción

Desde la raíz del proyecto:

```powershell
python Codigo/codim2_exacto.py                      # certificados.json (< 1 min)
$env:JULIA_DEPOT_PATH = Join-Path (Get-Location) 'Codigo/continuacion-julia/depot'
julia --startup-file=no --project=Codigo/continuacion-julia Codigo/codim2_hill_bk.jl 2 81
```

Entorno: SymPy 1.14.0, NumPy 2.4.4; Julia 1.12.4 y BifurcationKit 0.8.5
(entorno fijado en `Codigo/continuacion-julia`, sin instalación global).

## Contenido

- `certificados.json`:
  - identidades de la dicotomía del lazo y polinomio de Erlang (verificación simbólica, k <= 6);
  - familia de Hill ajustada: sigma2, sigma3 y el certificado de signo de 2*omega*l1(n)
    para n >= 9/10 en ambos Hopf;
  - sigma3* exacto (Hopf degenerado con sigma2 = -1/450);
  - ventanas de Erlang k = 1..8, que son raíces aisladas numéricamente, no certificadas.
- `curvas-hopf-n2-h81.csv`: curvas de Hopf en (Z, kappa), con omega y el coeficiente
  que registra BifurcationKit.
- `especiales-n2-h81.txt`: puntos especiales detectados. Se detectó GH en
  Z ~ 30.06939, kappa ~ 0.440940; no se detectaron BT ni ZH.

## Límites

- El valor de l2 que devuelve BifurcationKit usa derivadas de orden 4 y 5 por
  diferencias finitas. Las dos detecciones dieron signos opuestos (2.2e-3 y -3.9e-4),
  así que la no degeneración del punto GH queda indeterminada.
- El coeficiente l1 que registra BifurcationKit usa otra normalización. Solo su signo
  es comparable con el l1 del informe.
- El testigo de tres equilibrios con lazo positivo (AD-22) está certificado con
  aritmética racional en la sección del informe. No se continuaron sus pliegues ni su cúspide.

## Ampliación del 19/09/2026 (revisión del artículo)

Reproducción adicional, desde la raíz del proyecto:

```powershell
python Codigo/bautin_l2.py            # l2 en el punto GH por forma normal de orden 5 (60 dígitos)
python Codigo/bautin_simulacion.py    # control dinámico: pendiente de A^-4 en el punto GH
python Codigo/erlang_realizacion.py   # realización exacta de la ruptura de la dicotomía con k = 2
python Codigo/codim2_exacto.py        # añade el retardo discreto y Erlang k = 16, 32, 64
$env:JULIA_DEPOT_PATH = Join-Path (Get-Location) 'Codigo/continuacion-julia/depot'
julia --startup-file=no --project=Codigo/continuacion-julia Codigo/lpc_hill_bk.jl <Z>   # un valor de Z
python Codigo/figura_histeresis.py    # figura en inglés y en español
```

Resultados:
- `bautin-l2.json`:
  - GH en Z = 30.0693871235 y kappa = 0.4409398413, omega = 3.4719657057.
  - l1 de validación: reproduce el certificado del ejemplo racional.
  - l2 = +0.0171517 con v_x = 1 y +9.27e-4 con ||q|| = 1, estable con 40, 60 y 80 dígitos.
  - Residuo de grado 5: 3.5e-60.
  - Se sustituye el l2 indeterminado de BifurcationKit, obtenido por diferencias finitas.
- `bautin-simulacion.json`:
  - La pendiente de A_x^{-4} converge a -Re(G32)/48 = -0.01489 (-0.01406, -0.01460, -0.01473), lo que confirma l2 > 0.
- `lpc-resumen.csv` y `ciclos-Z*.csv`:
  - Pliegue global de ciclos grandes para 26 valores de Z.
  - Para Z >= 29, kappa_LPC < kappa_-: biestabilidad entre equilibrio estable y ciclo grande (histéresis).
  - Para 28.8 <= Z < Z_GH hay también un pliegue local de ciclos pequeños, por encima de kappa_-.
  - Los dos pliegues se funden en una cúspide de ciclos entre Z = 28.5 y 28.8.
  - Para Z = 70, 80 y 100, la columna kappa_menos del CSV es errónea: la continuación arrancó dentro de la ventana. Los valores correctos (0.1674 para Z = 80 y 0.1369 para Z = 100) los recalcula `figura_histeresis.py`.
- Estabilidad de la rama grande:
  - Las etiquetas `stable` de BifurcationKit fallan donde x(t) ~ 1e-5.
  - La estabilidad se verificó por integración variacional independiente: multiplicadores 0.30 a 0.84 en Z = 45, y 0.18 a 0.28 en Z = 35.
  - La figura usa esa verificación, no las etiquetas del CSV.
- `erlang-realizacion.json`:
  - Con k = 2, en kappa = 1: P_2 = lambda (4 lambda + 17)(100 lambda^2 + 1301)/400, es decir, pliegue--Hopf espectral.
  - Con g' = 99/100: C > 0, G' > 0, y cruces de Hopf en kappa = 0.43553 y 0.98990.
- `certificados.json`, clave `retardo_discreto`:
  - omega_c^2 = 231/50 + 3 sqrt(314454)/50, f'(omega_c) > 0, tau_c = 0.1091465, kappa_c = 9.16200.
- `histeresis-datos.csv`: datos de la figura.

Límites:
- l2 se obtiene con alta precisión, no con aritmética de intervalos.
- Los pliegues de ciclos provienen de continuación numérica.
- La realización con k = 2 es exacta, salvo la evaluación de autovalores a 30 dígitos en el caso perturbado.
