"""Experimentos reproducibles para la revisión del proyecto, modelo M2.

Ejecutar desde la raíz: python Codigo/proyecto_experimentos.py
Salidas exclusivas: Resultados/proyecto-revision, figures/es/pr-*.pdf
(con FIGURE_LANGUAGE=en: figures/pr-*.pdf, figura 1 del artículo),
Resultados/revision-proyecto/simulaciones.tex. No usa datos observacionales.
Las curvas de Hurwitz proceden de identidades; las órbitas se integran, no se
continúan con validación rigurosa. El refinamiento estima estabilidad numérica.
"""
from __future__ import annotations

import csv
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[1]
ENGLISH_FIGURES = os.environ.get("FIGURE_LANGUAGE", "es").lower() == "en"
OUT = ROOT / (Path("Resultados") / "figura1-en" if ENGLISH_FIGURES else Path("Resultados") / "proyecto-revision")
FIG = ROOT / (Path("figures") if ENGLISH_FIGURES else Path("figures") / "es")
FRAGMENT = ROOT / "Resultados" / "revision-proyecto"
for directory in [OUT, FIG, FRAGMENT]:
    directory.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 10, "axes.titlesize": 11, "legend.fontsize": 9,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "axes.unicode_minus": True,
})
BLUE, ORANGE, GRAY = "#24587a", "#b94b28", "#62686d"
KMINUS = (261 - np.sqrt(52921)) / 76
KPLUS = (261 + np.sqrt(52921)) / 76
ZCRIT = 77119 / 656
START = perf_counter()
records = []


def signal(x, y, cap=10.):
    return cap * (x + 8*y) / (81 + x + 8*y)


def field(t, u, speed, cap=10.):
    x, y, z = u
    return np.array([x*(12-x-y-10*z), y*(-2+4*x-y-z),
                     speed*(signal(x, y, cap)-z)])


def instant_field(t, u):
    x, y = u
    z = signal(x, y)
    return np.array([x*(12-x-y-10*z), y*(-2+4*x-y-z)])


def equilibrium(cap):
    # Rationalized smaller quadratic root: avoids subtractive cancellation.
    v = 787 + 337*cap
    z = 764*cap / (v + np.sqrt(v*v - 4*337*382*cap))
    return np.array([(14-9*z)/5, (46-41*z)/5, z])


def quantities(cap):
    x, y, z = equilibrium(cap)
    sx = 81*cap / (81+x+8*y)**2
    sy = 8*sx
    T, Q = x+y, 5*x*y
    B = T+10*x*sx+y*sy
    C = x*y*(5+9*sx+41*sy)
    L = T*B+Q-C
    disc = L*L-4*B*T*Q
    return x, y, z, sx, sy, T, Q, B, C, L, disc


def jacobian(u, speed, cap=10.):
    x, y, z = u
    sx = 81*cap / (81+x+8*y)**2
    return np.array([[12-2*x-y-10*z, -x, -10*x],
                     [4*y, -2+4*x-2*y-z, -y],
                     [speed*sx, 8*speed*sx, -speed]])


def save_csv(name, columns, rows):
    with (OUT/name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def figure_map():
    caps = np.linspace(.05, 125, 700)
    speeds = np.geomspace(.008, 20, 360)
    classes = np.full((len(speeds), len(caps)), 2, dtype=int)
    lower = np.full(caps.size, np.nan)
    upper = np.full(caps.size, np.nan)
    branch = []
    for j, cap in enumerate(caps):
        x,y,z,sx,sy,T,Q,B,C,L,disc = quantities(cap)
        if cap < ZCRIT:
            residual = np.linalg.norm(field(0, [x,y,z], 1, cap), ord=np.inf)
            assert residual < 2e-11 and min(x,y,z) > 0 and C > 0
            H = B*speeds**2+L*speeds+T*Q
            classes[:, j] = (H < 0).astype(int)
            if L < 0 and disc > 0:
                lower[j] = (-L-np.sqrt(disc))/(2*B)
                upper[j] = (-L+np.sqrt(disc))/(2*B)
            branch.append([cap,x,y,z,T,Q,B,C,L,disc,lower[j],upper[j],residual])
    # Independent spectral checks at fixed reproducible sample points.
    rng = np.random.default_rng(20260917)
    max_residual = 0.
    for _ in range(500):
        cap = rng.uniform(.05, ZCRIT-.01)
        speed = np.exp(rng.uniform(np.log(.008), np.log(20)))
        x,y,z,sx,sy,T,Q,B,C,L,disc = quantities(cap)
        J = jacobian([x,y,z], speed, cap)
        coefficients = np.poly(J)
        expected = np.array([1,T+speed,Q+B*speed,C*speed])
        max_residual = max(max_residual, float(np.max(np.abs(coefficients-expected))))
        h = B*speed**2+L*speed+T*Q
        if abs(h) > 1e-8:
            assert (max(np.linalg.eigvals(J).real) > 0) == (h < 0)
    assert max_residual < 1e-8
    births = []
    grid = np.linspace(.05, ZCRIT-.01, 1500)
    for a, b in zip(grid[:-1], grid[1:]):
        if quantities(a)[-1] * quantities(b)[-1] < 0:
            root = brentq(lambda z: quantities(z)[-1], a, b, xtol=1e-12)
            if quantities(root)[-2] < 0:
                births.append(root)
    assert len(births) == 2
    boundary_caps = births[0]+(births[1]-births[0])*(1-np.cos(np.linspace(0,np.pi,801)))/2
    boundary=[]
    for j,cap in enumerate(boundary_caps):
        x,y,z,sx,sy,T,Q,B,C,L,disc=quantities(cap)
        radical=0. if j in [0,len(boundary_caps)-1] else np.sqrt(max(0.,disc))
        boundary.append([cap,(-L-radical)/(2*B),(-L+radical)/(2*B)])
    boundary=np.array(boundary)
    save_csv("mapa-frontera-hurwitz.csv",["Zmax","kappa_inferior","kappa_superior"],boundary)
    save_csv("mapa-rama-equilibrio.csv",
             ["Zmax","x_star","y_star","z_star","T","Q","B","C","L",
              "discriminante","kappa_inferior","kappa_superior","residuo_equilibrio"], branch)
    save_csv("mapa-clasificacion.csv", ["Zmax","kappa","clase_0_estable_1_inestable_2_sin_interior"],
             ((cap, speed, classes[i,j]) for j,cap in enumerate(caps) for i,speed in enumerate(speeds)))
    fig, ax = plt.subplots(figsize=(6.65, 4.15))
    # Fill analytic boundaries rather than rasterizing the classification grid;
    # the resulting PDF contains only vector geometry and embedded text.
    ax.set_facecolor("#dce9ec")
    ax.axvspan(ZCRIT,125,color="#eeeeee")
    ax.fill_between(boundary[:,0],boundary[:,1],boundary[:,2],color="#f3c9b6")
    ax.plot(boundary[:,0],boundary[:,1],color=ORANGE,lw=1.7)
    ax.plot(boundary[:,0],boundary[:,2],color=ORANGE,lw=1.7)
    ax.axvline(ZCRIT, color=GRAY, ls="--", lw=1.2)
    ax.axvline(10, color=BLUE, ls=":", lw=1)
    ax.scatter([10,10], [KMINUS,KPLUS], color=BLUE, s=20, zorder=5)
    ax.annotate(r"$Z=10$", (10,.011), xytext=(15,.013), color=BLUE)
    ax.text(ZCRIT-1.7, .013, r"$Z_c=77119/656$", rotation=90, va="bottom", ha="right", color=GRAY, fontsize=9)
    ax.set(xlabel=(r"Regulatory target capacity $Z$" if ENGLISH_FIGURES else r"Capacidad del objetivo regulatorio $Z$"), ylabel=(r"Adjustment speed $\kappa$" if ENGLISH_FIGURES else r"Velocidad de ajuste $\kappa$"),
           yscale="log", xlim=(0,125), ylim=(.008,20))
    handles=[Patch(facecolor="#dce9ec", label=("Stable interior equilibrium" if ENGLISH_FIGURES else "Equilibrio interior estable")),
             Patch(facecolor="#f3c9b6", label=("Unstable interior equilibrium" if ENGLISH_FIGURES else "Equilibrio interior inestable")),
             Patch(facecolor="#eeeeee", label=("No interior equilibrium" if ENGLISH_FIGURES else "Sin equilibrio interior"))]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=.98)
    ax.grid(axis="y", alpha=.15)
    fig.tight_layout()
    fig.savefig(FIG/"pr-capacidad-estabilidad.pdf")
    fig.savefig(OUT/"pr-capacidad-estabilidad.png", dpi=140)
    plt.close(fig)
    return {"Z_coexistencia_exacto":"77119/656", "Z_coexistencia": ZCRIT,
            "extremos_capacidad_ventana_numericos": births,
            "malla_capacidades":len(caps),"malla_velocidades":len(speeds),
            "controles_espectrales":500,"max_residuo_coeficientes":max_residual,
            "semilla":20260917}


def hopf_data(branch):
    k = KMINUS if branch == "inferior" else KPLUS
    omega = np.sqrt(5+19*k/5)
    growth = -(7.6*k-26.1)/(2*((k+2)**2+omega**2))
    reG = (22213911154413348856042*k-440139298008754400484055)/431208247551462364945950
    v = (41+1j*omega)/(9+10j*omega)
    qvec = np.array([1,v,4-(1+1j*omega)*v])
    # x = 1 + w*q_x + conjugate(w*q_x), q_x=1:
    # A_x^2 / distance -> -8*abs(Re lambda')/Re G21.
    slope = -8*abs(growth)/reG
    return k,omega,growth,reG,qvec,slope


def event_max(t,u,speed,cap):
    return field(t,u,speed,cap)[0]
event_max.direction = -1


def event_min(t,u,speed,cap):
    return field(t,u,speed,cap)[0]
event_min.direction = 1


def periodic_run(branch, distance, refined=False):
    clock = perf_counter()
    kh,omega,growth,reG,qvec,slope = hopf_data(branch)
    speed = kh+distance if branch == "inferior" else kh-distance
    period = 2*np.pi/omega
    growth_actual = max(np.linalg.eigvals(jacobian([1,1,1], speed)).real)
    assert growth_actual > 0
    predicted_amp = np.sqrt(slope*distance)
    # An explicit common initial condition is used at both tolerances.
    initial = np.ones(3)+.45*predicted_amp*qvec.real
    assert min(initial) > 0
    burn = max(600., 9/growth_actual)
    tail_duration = 70*period
    rtol,atol = (2e-11,2e-13) if refined else (2e-9,2e-11)
    max_step = period/(24 if refined else 12)
    transient = solve_ivp(field, [0,burn], initial, args=(speed,10.),
                          method="DOP853",rtol=rtol,atol=atol,max_step=max_step)
    assert transient.success and np.min(transient.y) > 0
    tail = solve_ivp(field, [burn,burn+tail_duration], transient.y[:,-1], args=(speed,10.),
                     method="DOP853",rtol=rtol,atol=atol,max_step=max_step,
                     events=(event_max,event_min),dense_output=True)
    assert tail.success and np.min(tail.y) > 0
    maximum,minimum = tail.y_events[0][:,0],tail.y_events[1][:,0]
    assert min(len(maximum),len(minimum)) > 50
    amplitude = (np.median(maximum[-20:])-np.median(minimum[-20:]))/2
    amplitude_first = (np.median(maximum[:20])-np.median(minimum[:20]))/2
    measured_period = np.median(np.diff(tail.t_events[0][-21:]))
    drift = abs(amplitude-amplitude_first)/amplitude
    assert drift < 2e-5, (branch,distance,refined,drift)
    times = np.linspace(burn,burn+tail_duration,4201)
    values = tail.sol(times).T
    level = "refinado" if refined else "base"
    stem = f"{branch}-delta-{distance:g}-{level}"
    save_csv(f"orbita-{stem}.csv",["t","x","y","z"],np.column_stack([times,values]))
    save_csv(f"extremos-{stem}.csv",["t","x","y","z","tipo"],
             ([t,*u,kind] for kind,ts,us in zip(["maximo","minimo"],tail.t_events,tail.y_events)
              for t,u in zip(ts,us)))
    metric = {"rama":branch,"distancia_umbral":distance,"kappa":speed,
              "refinado":refined,"rtol":rtol,"atol":atol,"max_step":max_step,
              "dato_inicial":initial.tolist(),"t_descarte":burn,"t_ventana":tail_duration,
              "amplitud_x":amplitude,"periodo":measured_period,
              "deriva_relativa_amplitud":drift,"tasa_lineal_real":growth_actual,
              "minimo_componente":float(min(np.min(transient.y),np.min(tail.y))),
              "solver_exito":bool(transient.success and tail.success),
              "nfev":transient.nfev+tail.nfev,"segundos":perf_counter()-clock,
              "amplitud_predicha_primer_orden":predicted_amp}
    records.append(metric)
    print(json.dumps({key:metric[key] for key in ["rama","distancia_umbral","refinado","amplitud_x","periodo","deriva_relativa_amplitud","segundos"]}),flush=True)
    return metric,times,values,initial


def figure_local(experiments):
    fig, axes = plt.subplots(2,2,figsize=(6.75,5.9))
    controls = []
    for column,branch,distance in [(0,"inferior",.008),(1,"superior",.08)]:
        metric,times,values,initial = experiments[(branch,distance,"refinado")]
        tau = times-times[0]
        mask = tau <= 8*metric["periodo"]
        inst_times = np.linspace(0,25,1501)
        inst = solve_ivp(instant_field,[0,25],initial[:2],method="DOP853",rtol=2e-11,
                         atol=2e-13,max_step=.02,t_eval=inst_times)
        assert inst.success and min(inst.y.ravel()) > 0
        instant_distance = float(np.linalg.norm(inst.y[:,-1]-1))
        assert instant_distance < 1e-8
        save_csv(f"instantaneo-{branch}.csv",["t","x","y","z_objetivo"],
                 np.column_stack([inst.t,inst.y.T,signal(inst.y[0],inst.y[1])]))
        ax = axes[0,column]
        ax.plot(tau[mask],values[mask,0],color=ORANGE,lw=1.2,label=("Dynamic adjustment, final segment" if ENGLISH_FIGURES else "Ajuste dinámico, tramo final"))
        ax.axhline(1,color=GRAY,ls="--",lw=1.1,label=("Equilibrium / instantaneous limit" if ENGLISH_FIGURES else "Equilibrio / límite instantáneo"))
        ax.set(xlabel=(r"$\tau$ (final-segment time)" if ENGLISH_FIGURES else r"$\tau$ (tiempo del tramo final)"),ylabel=r"State $x$" if ENGLISH_FIGURES else r"Estado $x$",
               title=(f"{('Lower' if branch == 'inferior' else 'Upper') if ENGLISH_FIGURES else ('Umbral inferior' if branch == 'inferior' else 'Umbral superior')}: "+r"$\kappa=$"+f"{metric['kappa']:.5f}"))
        ax.tick_params(labelsize=9)
        ax = axes[1,column]
        onecycle = tau <= 2*metric["periodo"]
        ax.plot(values[onecycle,0],values[onecycle,1],color=ORANGE,lw=1.6,
                label=("Dynamic adjustment, observed cycle" if ENGLISH_FIGURES else "Ajuste dinámico, ciclo observado"))
        ax.plot(inst.y[0],inst.y[1],color=BLUE,ls="--",lw=1.1,
                label=("Instantaneous adjustment, transient" if ENGLISH_FIGURES else "Ajuste instantáneo, transitorio"))
        ax.scatter([1],[1],color=GRAY,s=19,zorder=5)
        ax.scatter([initial[0]],[initial[1]],facecolors="white",edgecolors=BLUE,s=26,zorder=5)
        ax.set(xlabel=(r"State $x$" if ENGLISH_FIGURES else r"Estado $x$"),ylabel=(r"State $y$" if ENGLISH_FIGURES else r"Estado $y$"))
        ax.grid(alpha=.16)
        controls.append({"rama":branch,"dato_inicial_xy":initial[:2].tolist(),
                         "modelo":"z=S(x,y), sistema reducido bidimensional",
                         "solver_exito":bool(inst.success),"rtol":2e-11,"atol":2e-13,
                         "t_final":25,"distancia_final_equilibrio":instant_distance})
    handles,labels=axes[0,0].get_legend_handles_labels()
    handles2,labels2=axes[1,0].get_legend_handles_labels()
    fig.legend(handles+handles2,labels+labels2,loc="lower center",ncol=2,
               fontsize=8,frameon=False,bbox_to_anchor=(.5,-.005))
    fig.tight_layout(rect=(0,.09,1,1))
    fig.savefig(FIG/"pr-hopf-comparacion-local.pdf")
    fig.savefig(OUT/"pr-hopf-comparacion-local.png",dpi=140)
    plt.close(fig)
    return controls


def figure_amplitude(experiments):
    fig,axes = plt.subplots(1,2,figsize=(6.75,3.25))
    diagnostics=[]
    rows=[]
    for ax,branch,distances in zip(axes,["inferior","superior"],
                                  [[.002,.004,.008,.016],[.02,.04,.08,.16]]):
        k,w,g,reG,qvec,slope = hopf_data(branch)
        amplitudes=[]
        for distance in distances:
            base = experiments[(branch,distance,"base")][0]
            fine = experiments[(branch,distance,"refinado")][0]
            relative_error = abs(base["amplitud_x"]-fine["amplitud_x"])/fine["amplitud_x"]
            period_error = abs(base["periodo"]-fine["periodo"])/fine["periodo"]
            assert relative_error < 2e-5 and period_error < 2e-5
            amplitudes.append(fine["amplitud_x"])
            rows.append([branch,distance,fine["kappa"],base["amplitud_x"],fine["amplitud_x"],
                         relative_error,base["periodo"],fine["periodo"],period_error,
                         fine["deriva_relativa_amplitud"],fine["t_descarte"],slope])
        amplitudes=np.array(amplitudes)
        domain=np.linspace(0,max(distances)*1.07,100)
        ax.plot(domain,slope*domain,color=GRAY,ls="--",lw=1.2,label=("Asymptotic Hopf prediction" if ENGLISH_FIGURES else "Predicción asintótica de Hopf"))
        ax.plot(distances,amplitudes**2,"o-",color=ORANGE,lw=1,ms=4,label=("Integrated, refined orbits" if ENGLISH_FIGURES else "Órbitas integradas, refinadas"))
        ax.set(xlabel=(r"Distance from threshold $\delta$" if ENGLISH_FIGURES else r"Distancia al umbral $\delta$"),ylabel=(r"Squared semi-amplitude $A_x^2$" if ENGLISH_FIGURES else r"Semiamplitud al cuadrado $A_x^2$"),
               title=(("Lower threshold" if branch == "inferior" else "Upper threshold") if ENGLISH_FIGURES else f"Umbral {branch}"),xlim=(0,max(domain)),ylim=(0,None))
        ax.grid(alpha=.17)
        fitted_exponent=np.polyfit(np.log(distances),np.log(amplitudes),1)[0]
        diagnostics.append({"rama":branch,"pendiente_asintotica_A2":slope,
                            "exponente_loglog_observado":float(fitted_exponent),
                            "max_error_relativo_refinamiento_amplitud":max(row[5] for row in rows if row[0]==branch),
                            "max_error_relativo_refinamiento_periodo":max(row[8] for row in rows if row[0]==branch)})
    axes[0].legend(loc="upper left",fontsize=7.4,frameon=False)
    fig.tight_layout()
    fig.savefig(FIG/"pr-amplitud-refinamiento.pdf")
    fig.savefig(OUT/"pr-amplitud-refinamiento.png",dpi=140)
    plt.close(fig)
    save_csv("amplitudes-refinamiento.csv",["rama","delta","kappa","A_base","A_refinada",
             "error_relativo_A","periodo_base","periodo_refinado","error_relativo_periodo",
             "deriva_relativa_A","t_descarte","pendiente_teorica_A2"],rows)
    return diagnostics


def write_protocol(summary):
    mapa=summary["mapa"]
    diag=summary["amplitud"]
    maxamp=max(row["max_error_relativo_refinamiento_amplitud"] for row in diag)
    maxperiod=max(row["max_error_relativo_refinamiento_periodo"] for row in diag)
    maxdrift=max(row["deriva_relativa_amplitud"] for row in records)
    burns=[row["t_descarte"] for row in records]
    def sci(v):
        exponent=int(np.floor(np.log10(v)))
        return rf"{v/10**exponent:.2f}\times10^{{{exponent}}}"
    text=rf"""% Generado por Codigo/proyecto_experimentos.py. No editar cifras a mano.
\subsection*{{Simulaciones de contraste para la formulación revisada}}

El protocolo examina tres preguntas: cómo cambia la ventana de inestabilidad al
variar la capacidad; si el ajuste instantáneo conserva las oscilaciones observadas
cerca de los umbrales; y si las amplitudes numéricas son compatibles con la escala
local de Hopf. Los parámetros son ilustrativos y adimensionales, no estimaciones
empíricas. Se fija $q=e=a=d=1$, $c=4$, $b=10$, $r=K=12$, $m=2$,
$\theta=1$, $\varphi=8$ y $h=81$. La capacidad $Z$ varía solo en la
figura~\ref{{fig:pr-capacidad}}; las otras dos usan $Z=10$.

\textbf{{Capacidad y estabilidad.}} Las nulclinas proporcionan
$x_*=(14-9z_*)/5$, $y_*=(46-41z_*)/5$ y
$337z_*^2-(787+337Z)z_*+382Z=0$. Se selecciona la raíz que pertenece
al intervalo de coexistencia. La frontera de existencia es exactamente
$Z_c=77119/656$. Para cada equilibrio se evalúa la cuadrática de Hurwitz,
utilizando las derivadas de la señal en ese punto; no se reutiliza el jacobiano
de $(1,1,1)$ cuando cambia la capacidad. Las curvas de la figura son raíces de
esa cuadrática y se contrastaron con 500 evaluaciones independientes del espectro.
El máximo residuo de los coeficientes fue ${sci(mapa['max_residuo_coeficientes'])}$.

En el barrido se detectó una ventana aproximadamente para
${mapa['extremos_capacidad_ventana_numericos'][0]:.4f}<Z<
{mapa['extremos_capacidad_ventana_numericos'][1]:.4f}$, con todos los demás
parámetros fijos. Estos dos extremos de capacidad se calcularon numéricamente;
no se confunden con el umbral exacto de existencia ni con un certificado de Hopf
no degenerado a lo largo de toda la curva. El resultado muestra que existencia
y estabilidad responden de manera distinta a la capacidad del regulador.

\begin{{figure}}[tbp]
\centering
\includegraphics[width=0.98\textwidth]{{figuras/pr-capacidad-estabilidad.pdf}}
\caption{{Mapa de coexistencia y estabilidad del equilibrio interior. Las curvas
delimitan la pérdida de la propiedad de Hurwitz; el área naranja no representa
una clasificación de todos los atractores. La región gris indica ausencia de
equilibrio interior, sin afirmar extinción global. La línea $Z=10$ identifica el
ejemplo con dos Hopf certificados.}}
\label{{fig:pr-capacidad}}
\end{{figure}}

\textbf{{Integración y control del transitorio.}} Se utilizó DOP853 de SciPy
{scipy.__version__}, con tolerancias base $2\times10^{{-9}}$ y
$2\times10^{{-11}}$ (relativa y absoluta). Cada caso se repitió con
$2\times10^{{-11}}$ y $2\times10^{{-13}}$, reduciendo además el paso máximo
de un doceavo a un veinticuatroavo del período lineal. El tiempo descartado se
fijó como $\max\{{600,9/\rho\}}$, donde $\rho>0$ es la parte real del
autovalor inestable: varió entre {min(burns):.1f} y {max(burns):.1f} unidades
adimensionales. Después se observaron 70 períodos lineales. Máximos y mínimos
de $x$ se localizaron mediante eventos de $\dot x=0$, con dirección del cruce;
la semiamplitud usa la diferencia entre las medianas de los últimos 20 máximos
y mínimos, dividida por dos. El período usa los últimos 20 intervalos entre
máximos. Todas las integraciones finalizaron correctamente y conservaron
coordenadas positivas en los puntos calculados.

\begin{{figure}}[tbp]
\centering
\includegraphics[width=0.98\textwidth]{{figuras/pr-hopf-comparacion-local.pdf}}
\caption{{Oscilaciones cerca de ambos Hopf para
$\kappa=\kappa_-+0.008$ (izquierda) y $\kappa=\kappa_+-0.08$ (derecha).
Arriba: ocho períodos del tramo final, con origen temporal desplazado después
del transitorio descartado. Abajo: proyección de la órbita tardía y transitorio
del modelo reducido $z=S(x,y)$, iniciado con los mismos valores de $x,y$.
El punto representa coexistencia y el círculo abierto el dato inicial del
modelo reducido. En esta comparación el ajuste instantáneo converge al
equilibrio; la conclusión corresponde a estos parámetros y datos iniciales.}}
\label{{fig:pr-local}}
\end{{figure}}

\textbf{{Amplitud local y refinamiento.}} Se integraron cuatro distancias
al interior de la ventana desde cada umbral:
\[
\begin{{array}}{{ll}}
\kappa=\kappa_-+\delta: & \delta\in\{{0.002,0.004,0.008,0.016\}},\\
\kappa=\kappa_+-\delta: & \delta\in\{{0.02,0.04,0.08,0.16\}}.
\end{{array}}
\]
El coeficiente
exacto de Hopf predice $A_x^2/\delta\to -8|\rho'|/\mathrm{{Re}}G_{{21}}$
cuando la primera componente del autovector derecho es 1. Los datos se
comparan con esa predicción, sin ajustar su pendiente. Los exponentes
logarítmicos observados fueron {diag[0]['exponente_loglog_observado']:.4f} y
{diag[1]['exponente_loglog_observado']:.4f}, respectivamente, en estos intervalos
finitos; no constituyen una demostración del exponente asintótico $1/2$.
El máximo cambio relativo de amplitud entre tolerancias fue ${sci(maxamp)}$;
para el período fue ${sci(maxperiod)}$. La máxima deriva relativa entre los
primeros y los últimos 20 extremos de una ventana fue ${sci(maxdrift)}$.
Estas comparaciones controlan sensibilidad numérica y transitorio, no ofrecen
cotas rigurosas de error ni sustituyen el teorema local.

\begin{{figure}}[tbp]
\centering
\includegraphics[width=0.98\textwidth]{{figuras/pr-amplitud-refinamiento.pdf}}
\caption{{Compatibilidad de las amplitudes con la predicción local de Hopf.
Los puntos corresponden a integraciones independientes y refinadas; las líneas
discontinuas usan coeficientes analíticos, sin ajuste a los puntos.
El barrido no es continuación numérica certificada y no demuestra que las dos
ramas periódicas se conecten.}}
\label{{fig:pr-amplitud}}
\end{{figure}}

\textbf{{Reproducibilidad y alcance.}} El programa
\texttt{{proyecto\_experimentos.py}}, en la carpeta \texttt{{Codigo}},
produce las figuras vectoriales,
los CSV de parámetros, mapas, órbitas y extremos, y el registro de tolerancias,
pasos máximos, tiempos de descarte, evaluaciones y éxito del solver.
Las salidas se guardan en \texttt{{proyecto-revision}}, dentro de
\texttt{{Resultados}}. Los resultados numéricos son
ilustraciones y controles de las pruebas del modelo M2; no estiman ciclos
económicos observados ni sustentan recomendaciones regulatorias.
"""
    (FRAGMENT/"simulaciones.tex").write_text(text,encoding="utf-8")
    report=["# Experimentos para revisión del proyecto", "",
            "Ejecutar: `python Codigo/proyecto_experimentos.py`.", "",
            "Modelo M2, parámetros adimensionales. No calibración ni datos empíricos.",
            "Las tres figuras se regeneran de los CSV y del resumen JSON.", "",
            "## Hallazgos", "",
            f"- Coexistencia: 0 < Z < 77119/656 = {ZCRIT:.12g}.",
            f"- Ventana espectral encontrada para capacidades entre {mapa['extremos_capacidad_ventana_numericos'][0]:.10g} y {mapa['extremos_capacidad_ventana_numericos'][1]:.10g} (extremos numéricos; búsqueda de ceros sobre la rama).",
            f"- Error relativo máximo de refinamiento de amplitud: {maxamp:.6g}; período: {maxperiod:.6g}.",
            f"- Deriva máxima en ventana: {maxdrift:.6g}.",
            f"- Exponentes observados log A/log delta: {diag[0]['exponente_loglog_observado']:.8g}, {diag[1]['exponente_loglog_observado']:.8g}.",
            "", "## Límites", "",
            "El mapa clasifica equilibrios; no identifica todos los atractores. Las curvas de Hurwitz fuera del ejemplo Z=10 no tienen aquí certificado de Lyapunov. Las amplitudes son resultados de integración con refinamiento, no continuación certificada, ni pruebas de conexión global. Un acuerdo entre tolerancias no es una cota rigurosa de error.",
            "", "## Artefactos", "",
            "- mapa-rama-equilibrio.csv: coordenadas, coeficientes y fronteras.",
            "- mapa-frontera-hurwitz.csv: trazado fino y cierre de curvas en los extremos numéricos.",
            "- mapa-clasificacion.csv: malla completa con clases explícitas.",
            "- orbita-*.csv y extremos-*.csv: ventanas y eventos, a ambas tolerancias.",
            "- amplitudes-refinamiento.csv: medidas y discrepancias por caso.",
            "- instantaneo-*.csv: comparación reducida con datos iniciales iguales en x,y.",
            "- resumen.json: versiones, parámetros, protocolo completo, éxito y coste por ejecución."]
    (OUT/"LEEME.md").write_text("\n".join(report)+"\n",encoding="utf-8")


if __name__ == "__main__":
    summary={"versiones":{"python":platform.python_version(),"numpy":np.__version__,
                          "scipy":scipy.__version__,"matplotlib":matplotlib.__version__},
             "parametros":{"r":12,"K":12,"q":1,"a":1,"b":10,"c":4,"d":1,
                           "e":1,"m":2,"theta":1,"phi":8,"h":81,"Zmax_orbitas":10},
             "solver":"DOP853","kappa_menos":KMINUS,"kappa_mas":KPLUS,
             "mapa":figure_map()}
    experiments={}
    for branch,distances in [("inferior",[.002,.004,.008,.016]),("superior",[.02,.04,.08,.16])]:
        for distance in distances:
            for refined in [False,True]:
                result=periodic_run(branch,distance,refined)
                experiments[(branch,distance,"refinado" if refined else "base")]=result
    summary["instantaneo"]=figure_local(experiments)
    summary["amplitud"]=figure_amplitude(experiments)
    summary["ejecuciones"]=records
    summary["tiempo_total_segundos"]=perf_counter()-START
    (OUT/"resumen.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    write_protocol(summary)
    print(json.dumps({"estado":"OK","segundos":summary["tiempo_total_segundos"],
                      "mapa":summary["mapa"],"amplitud":summary["amplitud"]},ensure_ascii=False,indent=2))
