# Code and data for "Timing-induced instability, memory, and hysteresis in adaptive nonlinear regulation"

Dolores Sanchez-Garcia, Diana Mercedes Castro-Cardenas, Segundo Francisco Segura-Altamirano,
Freddy Asrael Paz-Sifuentes and Santos Henry Guevara-Quiliche
(Universidad Nacional Pedro Ruiz Gallo, Peru).
Corresponding author: Segundo Francisco Segura-Altamirano, sseguraal@unprg.edu.pe.

Article submitted to *Chaos: An Interdisciplinary Journal of Nonlinear Science* (AIP Publishing).
Archive DOI: **[to be inserted after the Zenodo deposit]**.

This archive contains every script, parameter and numerical output needed to reproduce
the computational results, figures and tables of the article. No empirical data are used.

## The model

$$
\dot x=x(r-qx-ay-bz),\qquad
\dot y=y(-m+cx-ey-dz),\qquad
\dot z=\kappa\bigl(g(\theta x+\varphi y)-z\bigr),
$$

with a rational target $g(s)=Zs/(h+s)$ or a Hill target $g(s)=Zs^n/(h^n+s^n)$.
In Section 5 of the article the exponential memory of $z$ is replaced by an Erlang chain of $k$ stages with the same mean delay $1/\kappa$.

- **Example I:** $r=K=12$, $a=d=e=1$, $b=10$, $c=4$, $m=2$, $\theta=1$, $\varphi=8$, $h=81$, $Z=10$, with equilibrium $E_*=(1,1,1)$.
- **Hill target of Section 6:** the rates of Example I with $n=2$, $h=81$.

## Rigorous versus computational results

The article separates two kinds of result, and this archive follows the same separation:

- **Exact certificates:** SymPy rational or algebraic arithmetic, with assertions that fail if an identity does not hold. They support the proofs.
- **Computations:** high-precision arithmetic, numerical continuation and ODE integration. They support the statements labelled *Computational result*. None of them is an interval-arithmetic proof.

## Directory structure

```
.
├── README.md             this file
├── CITATION.cff          citation metadata
├── requirements.txt      pinned Python packages
├── Codigo/               all scripts (Python and Julia)
│   └── continuacion-julia/   Julia environment (Project.toml, Manifest.toml)
├── Resultados/           numerical outputs, one folder per topic
│   ├── limites-globales/     singular limits, Table I, guaranteed slow threshold
│   ├── codim2-hill/          Erlang memory, Bautin point, cycle folds, hysteresis, robustness
│   ├── continuacion-global/  periodic branch of Example I and its independent checks
│   └── extension-hopf/       exact Hopf certificates and jet bounds (Appendix B)
├── figures/              the figures of the article (PDF)
│   ├── es/                   Spanish-language versions written by some scripts (initially empty)
│   └── variants/             auxiliary variants written by some scripts (initially empty)
└── tables/               LaTeX source of Table I
```

Folder and script names are in Spanish, the working language of the project.
`Codigo` = code, `Resultados` = results.
The `README.md` inside each `Resultados/` subfolder is the original working log, also in Spanish. It records protocols, software versions and observed values. Some claim identifiers in these logs (AD-, CN-) and theorem numbers refer to the authors' internal project report, not to the article; the table below gives the correspondence with the article.

## Correspondence with the article

| Article                                                                                                   | Script(s)                                                                               | Output(s)                                                                                                                                                           | Type                                         |
| --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| Theorem 4, Example I (exact window $\kappa_\pm$); Proposition 16 (supercritical Hopf, $l_1$)              | `adaptativo_hopf.py`                                                                    | console assertions                                                                                                                                                  | exact + 60/100-digit check                   |
| Remark 15, Appendix B (jet polynomial, sign box, concave counterexample, matched Hill family)             | `lyapunov_clase.py`, `codim2_exacto.py`                                                 | `Resultados/extension-hopf/lyapunov-clase.json`, `Resultados/codim2-hill/certificados.json`                                                                         | exact                                        |
| Proposition 5, Corollary 6, Theorem 12(ii) (loop-sign identities, Erlang polynomial)                      | `codim2_exacto.py`                                                                      | `certificados.json`                                                                                                                                                 | exact (symbolic, $k\le6$)                    |
| Proposition 14 ($k=2$ breaks the dichotomy)                                                               | `erlang_realizacion.py`                                                                 | `Resultados/codim2-hill/erlang-realizacion.json`                                                                                                                    | exact                                        |
| Table II (Erlang windows, 60 000-point grid)                                                              | `erlang_malla.py`, `codim2_exacto.py`                                                   | `Resultados/codim2-hill/erlang-malla.json`                                                                                                                          | computation                                  |
| Section 4, guaranteed $\kappa_{\rm slow}$ on $\Xi_{0.05}$ and $\Xi_{0.5}$                                 | `umbral_lento.py`                                                                       | `Resultados/limites-globales/umbral-lento.json`                                                                                                                     | explicit bounds, outward rounding            |
| Table I (1152 orbits, quasi-extinction), Fig. 2                                                           | `limites_globales.py`, then `tabla_limites.py`                                          | `convergencia-global-64.csv`, `resumen-64.json`, `tables/tabla-limits.tex`, `figures/limits-global.pdf`                                                             | computation (seed 20260918)                  |
| Fig. 1 (capacity-speed classification)                                                                    | `proyecto_experimentos.py` (with `FIGURE_LANGUAGE=en`)                                  | `figures/pr-capacidad-estabilidad.pdf`                                                                                                                              | computation                                  |
| Computational result 17 (Bautin point, $l_2>0$)                                                           | `bautin_l2.py`, `bautin_simulacion.py`                                                  | `Resultados/codim2-hill/bautin-l2.json`, `bautin-simulacion.json`                                                                                                   | 60-digit normal form + dynamical check       |
| Computational result 18 (fold of large-amplitude cycles, hysteresis), Fig. 3                              | `codim2_hill_bk.jl`, `lpc_hill_bk.jl`, `floquet_rama_grande.py`, `figura_histeresis.py` | `curvas-hopf-n2-h81.csv`, `especiales-n2-h81.txt`, `ciclos-Z*.csv`, `lpc-resumen.csv`, `floquet-rama-grande.json`, `histeresis-datos.csv`, `figures/hysteresis.pdf` | continuation + independent Floquet check     |
| Fig. 4 (quasistatic sweeps), Computational result 19 and Table III (robustness to ±5 % in $b$, $\varphi$) | `histeresis_robustez.py`                                                                | `barrido-kappa-Z45.csv`, `histeresis-robustez.json`, `figures/sweep.pdf`                                                                                            | computation                                  |
| Appendix C, Fig. 5 (periodic branch of Example I, 402 orbits)                                             | `continuacion_bk.jl` (three runs), then `validar_continuacion.py`                       | `Resultados/continuacion-global/*`                                                                                                                                  | continuation + independent IVP/Floquet check |

## Software

- **Python 3.14.4** with the packages pinned in `requirements.txt`.
- **Julia 1.12.4** with BifurcationKit 0.8.5. The environment is pinned in `Codigo/continuacion-julia/Manifest.toml`.

Operating system used: Windows 11.

```bash
python -m pip install -r requirements.txt
julia --project=Codigo/continuacion-julia -e "using Pkg; Pkg.instantiate()"
```

Optionally, set `JULIA_DEPOT_PATH` to a local folder, for example `Codigo/continuacion-julia/depot`, so that Julia packages are not installed globally. The depot itself is not archived.

## Reproduction

Run every command from the root of this archive. Each script overwrites its own outputs.
All Python scripts below, except the full `limites_globales.py` run, were run from this directory structure on 19/09/2026 and completed without errors, reproducing the stored outputs. Times were measured on a Windows laptop.

### 1. Exact certificates (under 1 min; `erlang_malla.py` about 45 s)

```bash
python Codigo/adaptativo_hopf.py
python Codigo/lyapunov_clase.py
python Codigo/codim2_exacto.py
python Codigo/erlang_realizacion.py
python Codigo/erlang_malla.py
```

### 2. Singular limits and Table I

```bash
python Codigo/umbral_lento.py            # about 3 min
python Codigo/limites_globales.py        # 1152 orbits; the longest step (not re-timed)
python Codigo/limites_globales.py --solo-figura   # figure only, from the stored runs
python Codigo/tabla_limites.py           # writes tables/tabla-limits.tex
```

### 3. Figure 1

```bash
FIGURE_LANGUAGE=en python Codigo/proyecto_experimentos.py
```

In PowerShell: `$env:FIGURE_LANGUAGE="en"; python Codigo/proyecto_experimentos.py`.
The script also writes two auxiliary figures that are not in the article (`figures/pr-amplitud-refinamiento.pdf` and `figures/pr-hopf-comparacion-local.pdf`), plus its tables in `Resultados/figura1-en/`.

### 4. Bautin point, folds of cycles and hysteresis (Section 6)

```bash
python Codigo/bautin_l2.py
python Codigo/bautin_simulacion.py       # about 4 min
julia --project=Codigo/continuacion-julia Codigo/codim2_hill_bk.jl 2 81
# one continuation per capacity value; lpc-resumen.csv is appended by each run
for Z in 27 27.5 28 28.5 28.8 29 29.2 29.5 29.8 30 30.5 30.6 31 32 33 34 35 37 40 42 45 50 55 60 70 80 100; do
  julia --project=Codigo/continuacion-julia Codigo/lpc_hill_bk.jl $Z
done
python Codigo/floquet_rama_grande.py     # about 1 min; independent Floquet check, also recomputes kappa_- exactly
python Codigo/figura_histeresis.py       # Fig. 3
python Codigo/histeresis_robustez.py     # about 9 min; Fig. 4 and Table III
```

Delete `Resultados/codim2-hill/lpc-resumen.csv` before rerunning the loop, because each run appends a row to it.

### 5. Periodic branch of Example I (Appendix C)

```bash
julia --project=Codigo/continuacion-julia Codigo/continuacion_bk.jl 40 lower
julia --project=Codigo/continuacion-julia Codigo/continuacion_bk.jl 80 lower
julia --project=Codigo/continuacion-julia Codigo/continuacion_bk.jl 80 upper
python Codigo/validar_continuacion.py    # Fig. 5
```

The Julia continuations were not re-run for this archive. Their stored outputs are the ones used in the article.

## Known numerical caveats

These are also stated in the article.

- **Floquet stability labels.** BifurcationKit's collocation Floquet solver misclassifies parts of the large-amplitude branch where $x(t)$ comes within $10^{-5}$ of zero. Stability of that branch is taken from the independent variational integration in `floquet-rama-grande.json`, not from the `stable` column of `ciclos-Z*.csv`.
- **Sign of $l_2$.** An earlier finite-difference estimate of $l_2$ in the continuation software gave both signs and was discarded. The value reported in the article comes from `bautin_l2.py`.
- **Table II for $k\geq4$.** The rows rest on a 60 000-point grid, which does not exclude additional stability switches rigorously.
- **Sweeps.** The quasistatic sweeps use a finite integration time per step, so the onset on the increasing sweep lags $\kappa_-$ slightly.

## Licence

Source code is released under the MIT License. Numerical data, generated datasets, tables, and figures produced for this study are released under the Creative Commons Attribution 4.0 International License (CC BY 4.0). Third-party materials remain subject to their original licenses.

## How to cite

Please cite the article and this archive. See `CITATION.cff`.
