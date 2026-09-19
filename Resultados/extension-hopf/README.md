# Dependencia del primer coeficiente de Lyapunov

Fecha: 18/09/2026. Modalidad T con cálculo algebraico exacto y controles C.
Afirmaciones AD-8, AD-9 y AD-10.

## Reproducción y evidencia

Desde la raíz: `python Codigo/lyapunov_clase.py` .
Requiere SymPy, NumPy y SciPy. lyapunov-clase.json registra versiones,
coeficientes, cotas racionales exactas y ejemplos numéricos.
La ejecución se completó sin fallos de aserción.
Las dependencias Python usadas se fijan en Codigo/requirements-extension.txt
(Python 3.14.4; SymPy 1.14.0).

Las demostraciones se encuentran en MBE_template/extension-lyapunov-en.tex
y en informe/extension-lyapunov.tex. La fórmula normal utiliza la convención
de Kuznetsov, ecuación (5.62), con autovector v_x=1 y l1=Re(G21)/(2 omega).

## Resultado general

Fijados equilibrio, parámetros, valor y pendiente del objetivo regulatorio
g en la señal de equilibrio s*, la dependencia respecto de sus derivadas es

2 omega l1 = a0 + a1 g''(s*) + a2 [g''(s*)]^2 + a3 g'''(s*).

La razón es que la derivada bilineal depende de modo afín de g'' y la
trilineal depende linealmente de g'''; el jacobiano y sus resolventes
permanecen fijos. Esta identidad no impone por sí sola un signo universal.

## Condición suficiente cuantificada

Para las tasas del ejemplo, g(9)=1 y g'(9)=1/10 conservan el equilibrio
(1,1,1) y los Hopf kappa=(261 +/- sqrt(52921))/76.
Se calculan los cuatro coeficientes en el cuerpo
Q[t]/(10t^4+361t^2+1935), t=i omega, con conjugación t -> -t.

Para cualquier respuesta C3, creciente, cóncava y acotada, con g(0)=0,
los intervalos

- -1/300 <= g''(9) <= -1/900;
- -1/13500 <= g'''(9) <= 1/4500

garantizan dos Hopf supercríticos no degenerados bajo esas condiciones
de coincidencia. Las cotas exactas dan Re(G21)<-0.6264 en el inferior
y <-0.6196 en el superior. Las cifras mostradas relajan hacia afuera
las fracciones guardadas en JSON.

El script reconstruye exactamente el certificado anterior de la respuesta
racional y compara evaluaciones con otro cálculo de autovectores y
resolventes en doble precisión. Este último control no reemplaza el
cálculo racional ni constituye verificación formal en un asistente de pruebas.

## Límite de la concavidad

Una perturbación suave compactamente soportada alrededor de s=9 preserva
valor, pendiente y segunda derivada, pero puede fijar g'''(9)=1/100.
Para soporte suficientemente pequeño conserva incremento, concavidad
estricta y saturación. Las cotas exactas resultantes son Re(G21)>6.5624
y >1.6871: ambos Hopf pasan a ser subcríticos.

Por tanto, la concavidad sola no garantiza l1<0. Se mantienen la ventana
lineal y los umbrales, pero cambia la criticidad. Este resultado negativo
delimita la generalización en vez de afirmar supercriticidad universal.

Se incluyen cinco ejemplos de respuestas ajustadas: cuatro algebraicas
(p=1/2,1,2,4) y una exponencial. Sus parámetros y coeficientes son
evaluaciones numéricas; los decimales de la tabla no están certificados.

La condición suficiente varía dos derivadas manteniendo valor y pendiente.
No es una caja de variación simultánea de todas las tasas del modelo.
La persistencia frente a otras perturbaciones es local y cualitativa.
Las demostraciones y la reducción algebraica quedan disponibles para
revisión humana; no se atribuye aprobación a revisores externos.
