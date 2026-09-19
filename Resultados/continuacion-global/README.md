# Continuación numérica de ciclos y controles independientes

Fecha: 18/09/2026. Modalidad C; afirmación CN-3.
Caso: M2 racional, parámetros (11) del manuscrito, equilibrio (1,1,1).
Esta carpeta contiene cálculos realizados, no una certificación global.

## Reproducción

Desde la raíz del proyecto, con Julia 1.12.4 disponible en PATH:

```powershell
$env:JULIA_DEPOT_PATH = Join-Path (Get-Location) 'Codigo/continuacion-julia/depot'
julia --startup-file=no --project=Codigo/continuacion-julia -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=Codigo/continuacion-julia Codigo/continuacion_bk.jl 40 lower
julia --startup-file=no --project=Codigo/continuacion-julia Codigo/continuacion_bk.jl 80 lower
julia --startup-file=no --project=Codigo/continuacion-julia Codigo/continuacion_bk.jl 80 upper
python Codigo/validar_continuacion.py
```

La primera orden de Julia restaura el entorno; requiere acceso a red en una
máquina sin las dependencias. Project.toml y Manifest.toml fijan el entorno.
El depot es local al proyecto, no una instalación global. Las ejecuciones
reescriben sus artefactos derivados con los mismos nombres.

Entorno ejecutado: Julia 1.12.4, BifurcationKit 0.8.5, Python 3.14.4,
NumPy 2.4.4, SciPy 1.17.1. El validador usa también Matplotlib.
Matplotlib fue 3.10.8; las dependencias se enumeran en
Codigo/requirements-extension.txt.
Los hashes SHA-256 de los dos scripts y del Manifest están en verification.json.
No hay muestreo aleatorio ni semilla.

## Protocolo y archivos

- PALC con tangente secante, colocación ortogonal de grado 4 y mallas
  uniformes de 40 y 80 intervalos; Newton 1e-10; paso máximo 0.04.
- Cambio de rama desde cada Hopf, desplazamiento inicial aproximado 0.002.
  Parada a distancia 1e-6 del Hopf opuesto.
- branch-lower-n40.csv: 135 órbitas; branch-lower-n80.csv: 133;
  branch-upper-n80.csv: 134. Los .jls conservan las ramas serializadas;
  su lectura requiere el entorno Julia correspondiente.
- Los CSV guardan período, punto inicial, extremos muestreados en 2001
  instantes de la interpolación de colocación, residual sin la ecuación
  de fase y logaritmos de los multiplicadores calculados por BifurcationKit.
- run-*.txt conserva configuración, tiempo de ejecución y puntos especiales.
- independent-checks.csv guarda el cierre y los multiplicadores calculados
  integrando la EDO original y su ecuación variacional con DOP853:
  rtol=2e-12, atol=2e-14, paso máximo igual al período/80.
- Se corrigen por shooting 14 puntos por ejecución y se comparan ambos
  sentidos en 21 velocidades comunes tras corrección de fase.
  bidirectional-common-speeds.csv contiene esta segunda comparación.
- verification.json conserva los resúmenes y límites de interpretación.

## Resultados observados

El sentido creciente cubre [0.4092943584, 6.4611259427]; el decreciente,
[0.4072951099, 6.4591274138]. No se detectan pliegues ni cambios secundarios
de estabilidad en estas ramas calculadas. Los dos multiplicadores no
triviales tienen módulo menor que uno en las 402 órbitas guardadas.

| Control | Máximo observado, redondeado hacia arriba |
| --- | ---: |
| Residual de colocación y frontera periódica | 9.0e-11 |
| Cierre independiente en norma infinito | 3.2e-9 |
| Diferencia del multiplicador neutro respecto a uno | 1.9e-8 |
| Diferencia de multiplicadores colocación/EDO | 2.2e-8 |
| Diferencia de períodos entre sentidos a velocidad común | 2.3e-12 |
| Diferencia de extremos entre sentidos a velocidad común | 2.7e-11 |

El mayor módulo no trivial es aproximadamente 0.999999737 cerca del Hopf
superior. La semiamplitud máxima muestreada de x es aproximadamente 2.498;
la coordenada mínima muestreada es aproximadamente 0.02457.
Los extremos de colocación se obtienen por muestreo, no por una cota
certificada del máximo continuo.

## Figuras y alcance

El validador genera la figura inglesa en
MBE_template/figures-en/global-cycle-continuation.pdf y la española en
proyecto/figuras/global-cycle-continuation.pdf. Se integraron en las
secciones 8.1 del artículo y 3.10 del informe.

La concordancia entre mallas, sentidos y métodos respalda una rama
periódica estable que se aproxima a ambos Hopf. No demuestra existencia
para cada valor del intervalo continuo, unicidad global, ausencia de
ramas desconectadas ni cuencas de atracción. El residual pequeño no es
una cota rigurosa del error. No se infiere validación económica.

## Fuentes técnicas consultadas

- Documentación oficial, formulación por colocación y continuación:
  https://bifurcationkit.github.io/BifurcationKitDocs.jl/stable/periodicOrbitCollocation/
- Código, pruebas y CITATION.bib de BifurcationKit 0.8.5, disponibles en
  el entorno local y en https://github.com/bifurcationkit/BifurcationKit.jl .
- Referencia bibliográfica añadida: R. Veltz, BifurcationKit.jl (2020),
  https://hal.science/hal-02902346 .

No se ejecutó AUTO. La elección implementada fue BifurcationKit.
