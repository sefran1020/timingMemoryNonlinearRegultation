"""Signo de la abscisa espectral del ejemplo I con memoria de Erlang en una malla fina.

Usa los autovalores del jacobiano completo (k+2)x(k+2), mejor condicionado que las raíces
del polinomio de grado k+2. Respalda la tabla de ventanas de Erlang del artículo.
Uso: python Codigo/erlang_malla.py   Salida: Resultados/codim2-hill/erlang-malla.json
"""
import numpy as np
from numpy.polynomial import polynomial as P
# Example I at E*=(1,1,1)
q,a,b,c,e,d,al,be=1,1,10,4,1,1,0.1,0.8
def jac(k,kap):
    n=k+2; J=np.zeros((n,n))
    J[0,:2]=[-q,-a]; J[0,n-1]=-b
    J[1,:2]=[c,-e]; J[1,n-1]=-d
    J[2,0],J[2,1],J[2,2]=k*kap*al,k*kap*be,-k*kap
    for j in range(3,n): J[j,j-1],J[j,j]=k*kap,-k*kap
    return J
T,Q,B,C=2,5,3.8,38.7
def ab_poly(k,kap):
    p=P.polyadd(P.polymul([Q,T,1.0],P.polypow([k*kap,1.0],k)),(k*kap)**k*np.array([C-Q,B-T]))
    return np.max(np.roots(p[::-1]).real)
import json
from pathlib import Path
out={}
for k in (4,8,16,64):
    up={4:8.76972,8:8.98808,16:9.08059,64:9.14269}[k]
    ks=np.concatenate([np.logspace(-6,np.log10(up),60001)])
    ab=np.array([np.max(np.linalg.eigvals(jac(k,x)).real) for x in ks])
    # check consistency with polynomial at a few points
    chk=max(abs(ab[i]-ab_poly(k,ks[i])) for i in range(0,60001,6000)) if k<=16 else None
    inside=ks<up-1e-4
    print(k,'min abscissa on (1e-6, up-1e-4):',ab[inside].min(),'positive everywhere:',bool((ab[inside]>0).all()),
          'n pts',inside.sum(),'poly-vs-jac diff',chk)
    # also just above up
    fuera=float(np.max(np.linalg.eigvals(jac(k,up+1e-3)).real))
    print("   abscissa at up+1e-3:",fuera)
    out[k]=dict(kappa_sup=up,malla=[1e-6,up-1e-4],puntos=int(inside.sum()),abscisa_min=float(ab[inside].min()),positiva_en_toda_la_malla=bool((ab[inside]>0).all()),abscisa_en_sup_mas_1e_3=fuera)
Path(__file__).resolve().parents[1].joinpath("Resultados","codim2-hill","erlang-malla.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
