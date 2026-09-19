"""Derivative-based Hopf coefficient and exact sign boxes for M2.

Run: python Codigo/lyapunov_clase.py
Uses q_x=1 and <p,q>=1. All coefficient identities and box bounds use
rational arithmetic in Q[t]/(10t^4+361t^2+1935), t=i*omega.
The box varies g''(9), g'''(9), keeping g(9)=1 and g'(9)=1/10,
where g=Z*f is the implemented target as a function of the signal.
"""
import json
from pathlib import Path
import sympy as sp
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'Resultados' / 'extension-hopf'


def exact_coefficients():
    t, kr = sp.symbols('t kappa')
    field = sp.QQ.alg_field_from_poly(sp.Poly(10*t**4+361*t**2+1935,t))
    one, zero, gen = field.one, field.zero, field([1,0])
    k = -(5*gen**2+25)/19
    def conj(a):
        cs=a.to_list(); n=len(cs)-1
        return field([v*(-1)**(n-i) for i,v in enumerate(cs)])
    def solve(matrix,rhs):
        rows=[[field.convert(v) for v in row]+[field.convert(rhs[i])]
              for i,row in enumerate(matrix)]
        for col in range(len(rows)):
            pivot=next(i for i in range(col,len(rows)) if rows[i][col])
            rows[col],rows[pivot]=rows[pivot],rows[col]
            scale=rows[col][col]; rows[col]=[v/scale for v in rows[col]]
            for i in range(len(rows)):
                if i != col:
                    a=rows[i][col]
                    rows[i]=[v-a*w for v,w in zip(rows[i],rows[col])]
        return [r[-1] for r in rows]
    def bilinear(u,v,s2):
        return [-2*u[0]*v[0]-u[0]*v[1]-u[1]*v[0]-10*(u[0]*v[2]+u[2]*v[0]),
                4*(u[0]*v[1]+u[1]*v[0])-2*u[1]*v[1]-u[1]*v[2]-u[2]*v[1],
                k*s2*(u[0]+8*u[1])*(v[0]+8*v[1])]
    J=[[-one,-one,-10*one],[4*one,-one,-one],[k/10,4*k/5,-k]]
    qy=(41+gen)/(9+10*gen)
    q=[one,qy,4-(1+gen)*qy]; qb=[conj(v) for v in q]
    ell=solve([[-one-gen,4*one],[-one,-one-gen]],[-k/10,-4*k/5])+[one]
    scale=sum((u*v for u,v in zip(ell,q)),zero)
    ell=[v/scale for v in ell]
    assert sum((v*w for v,w in zip(ell,q)),zero)==one
    resolvent=[[2*gen*(i==j)-J[i][j] for j in range(3)] for i in range(3)]
    def G(s2,s3):
        h11=solve(J,bilinear(q,qb,s2)); h20=solve(resolvent,bilinear(q,q,s2))
        C=[zero,zero,k*s3*(q[0]+8*q[1])**2*(qb[0]+8*qb[1])]
        b11=bilinear(q,h11,s2); b20=bilinear(qb,h20,s2)
        g=sum((ell[i]*(C[i]-2*b11[i]+b20[i]) for i in range(3)),zero)
        return (g+conj(g))/2
    g0,gp,gm,g3=G(0,0),G(1,0),G(-1,0),G(0,1)
    coefficients=[g0,(gp-gm)/2,(gp+gm)/2-g0,g3-g0]
    def express(a):
        cs=a.to_list()
        expr=sum(sp.Rational(v)*t**(len(cs)-1-i) for i,v in enumerate(cs))
        assert sp.Poly(expr,t).degree()<=2 and expr.coeff(t,1)==0
        return sp.factor(expr.subs(t**2,-5-sp.Rational(19,5)*kr))
    expressions=[express(v) for v in coefficients]
    # Independent reconstruction of the original rational target certificate.
    rational=sum(v*w for v,w in zip(expressions,[1,-sp.Rational(1,450),sp.Rational(1,450)**2,sp.Rational(1,13500)]))
    known=(22213911154413348856042*kr-440139298008754400484055)/sp.Integer(431208247551462364945950)
    assert sp.simplify(rational-known)==0
    assert G(sp.Rational(-1,200),sp.Rational(1,1000)) == sum(
        (v*field.convert(w) for v,w in zip(coefficients,[1,-sp.Rational(1,200),sp.Rational(1,200)**2,sp.Rational(1,1000)])),zero)
    return kr,expressions


def interval_add(a,b): return (a[0]+b[0],a[1]+b[1])
def interval_mul(a,b):
    p=[u*v for u in a for v in b]
    return min(p),max(p)


def independent_l1(k,s2,s3):
    """Double precision independent eigenvector/tensor evaluation."""
    J=np.array([[-1,-1,-10],[4,-1,-1],[k/10,4*k/5,-k]])
    eig,vec=np.linalg.eig(J)
    i=np.argmax(eig.imag); w=eig[i].imag
    q=vec[:,i]/vec[0,i]
    ev,pv=np.linalg.eig(J.T)
    p=pv[:,np.argmin(abs(ev+1j*w))]
    p=p/np.conj(np.vdot(p,q))
    def B(u,v):
        return np.array([-2*u[0]*v[0]-u[0]*v[1]-u[1]*v[0]-10*(u[0]*v[2]+u[2]*v[0]),
                         4*(u[0]*v[1]+u[1]*v[0])-2*u[1]*v[1]-u[1]*v[2]-u[2]*v[1],
                         k*s2*(u[0]+8*u[1])*(v[0]+8*v[1])])
    qb=q.conj()
    C=np.array([0,0,k*s3*(q[0]+8*q[1])**2*(qb[0]+8*qb[1])])
    value=C-2*B(q,np.linalg.solve(J,B(q,qb)))+B(qb,np.linalg.solve(2j*w*np.eye(3)-J,B(q,q)))
    return float(np.vdot(p,value).real/(2*w))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    k,cs=exact_coefficients()
    intervals=[(sp.Rational(407294109,10**9),sp.Rational(407294111,10**9)),
               (sp.Rational(6461126942,10**9),sp.Rational(6461126944,10**9))]
    s2=(sp.Rational(-1,300),sp.Rational(-1,900))
    s3=(sp.Rational(-1,13500),sp.Rational(1,4500))
    rows=[]
    for branch,krange in zip(['lower','upper'],intervals):
        H=lambda v:sp.Rational(19,5)*v*v-sp.Rational(261,10)*v+10
        assert H(krange[0])*H(krange[1])<0
        cranges=[tuple(sorted(c.subs(k,v) for v in krange)) for c in cs]
        square=(min(s2[0]**2,s2[1]**2),max(s2[0]**2,s2[1]**2))
        bound=cranges[0]
        for coeff,power in zip(cranges[1:],[s2,square,s3]):
            bound=interval_add(bound,interval_mul(coeff,power))
        assert bound[1]<0
        assert cranges[3][0]*cranges[3][1]>0
        kmid=sum(krange)/2
        cv=[float(c.subs(k,kmid)) for c in cs]
        w=float(sp.sqrt(5+sp.Rational(19,5)*kmid))
        rows.append(dict(branch=branch,kappa_interval=[str(v) for v in krange],
                         ReG_coefficients=cv,l1_coefficients=[v/(2*w) for v in cv],
                         ReG_box_upper_exact=str(bound[1]),ReG_box_upper=float(bound[1]),
                         l1_rational=(cv[0]-cv[1]/450+cv[2]/450**2+cv[3]/13500)/(2*w)))
        positive=(cranges[0][0]+cranges[1][1]*(-sp.Rational(1,450))
                  +cranges[2][0]*sp.Rational(1,450)**2+cranges[3][0]/100)
        assert positive>0
        rows[-1]['ReG_subcritical_lower_exact']=str(positive)
        rows[-1]['ReG_subcritical_lower']=float(positive)
        for d2,d3 in [(-1/450,1/13500),(-.003,.0002),(-1/450,.01)]:
            exact=float((cs[0]+cs[1]*d2+cs[2]*d2*d2+cs[3]*d3).subs(k,kmid))/(2*w)
            assert abs(exact-independent_l1(float(kmid),d2,d3))<2e-8
    families=[]
    for power in [.5,1.,2.,4.,np.inf]:
        if np.isinf(power):
            t=brentq(lambda t:t/np.expm1(t)-.9,1e-6,2.)
            h=9/t; cap=1/(-np.expm1(-t))
            d2=-.1/h; d3=.1/h**2
            name='exponential'
        else:
            t=brentq(lambda t:power*t/((1+t)*np.expm1(power*np.log1p(t)))-.9,1e-6,5.)
            h=9/t; cap=1/(-np.expm1(-power*np.log1p(t)))
            d2=-.1*(power+1)/(h+9); d3=.1*(power+1)*(power+2)/(h+9)**2
            name=f'algebraic p={power:g}'
        assert float(s2[0])<=d2<=float(s2[1]) and float(s3[0])<=d3<=float(s3[1])
        lvals=[independent_l1(float(sum(v)/2),d2,d3) for v in intervals]
        families.append(dict(family=name,h=h,capacity=cap,g2=d2,g3=d3,l1_lower=lvals[0],l1_upper=lvals[1]))
    summary=dict(sympy=sp.__version__,numpy=np.__version__,
                 normalization='q_x=1; <p,q>=1; l1=ReG21/(2*omega)',
                 ReG_coefficient_expressions=[str(c) for c in cs],
                 derivative_box=dict(g2=[str(v) for v in s2],g3=[str(v) for v in s3]),
                 branches=rows,matched_response_examples=families,exact_certificate=True)
    (OUT/'lyapunov-clase.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
