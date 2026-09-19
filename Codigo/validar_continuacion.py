"""Independent IVP/Floquet and shooting checks of BifurcationKit branches.

Does not use time integration to draw the continuation branch. The plotted
points come from BK collocation; SciPy DOP853 checks closure and multipliers.
Run after the three continuacion_bk.jl runs: python Codigo/validar_continuacion.py
"""
import csv
import json
from pathlib import Path
import platform
import hashlib
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import root, minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Resultados'/'continuacion-global'
KM=(261-np.sqrt(52921))/76
KP=(261+np.sqrt(52921))/76


def field(u,k):
    x,y,z=u
    return np.array([x*(12-x-y-10*z),y*(-2+4*x-y-z),k*(10*(x+8*y)/(81+x+8*y)-z)])


def jac(u,k):
    x,y,z=u; a=810/(81+x+8*y)**2
    return np.array([[12-2*x-y-10*z,-x,-10*x],[4*y,-2+4*x-2*y-z,-y],[k*a,8*k*a,-k]])


def flow(u0,T,k):
    def rhs(t,v): return np.r_[field(v[:3],k),(jac(v[:3],k)@v[3:].reshape(3,3)).ravel()]
    sol=solve_ivp(rhs,[0,T],np.r_[u0,np.eye(3).ravel()],method='DOP853',rtol=2e-12,atol=2e-14,
                  max_step=T/80,dense_output=True)
    assert sol.success
    return sol


def multipliers(sol):
    mu=np.linalg.eigvals(sol.y[3:,-1].reshape(3,3))
    neutral=np.argmin(abs(mu-1))
    return mu[neutral],np.delete(mu,neutral)


def extrema(sol,T):
    times=np.linspace(0,T,1001); values=sol.sol(times)[:3]
    answer=[]
    for j in range(3):
        for sign in [1,-1]:
            i=np.argmin(sign*values[j]); left=times[max(0,i-1)]; right=times[min(1000,i+1)]
            op=minimize_scalar(lambda t:sign*sol.sol(t)[j],bounds=(left,right),method='bounded',
                               options={'xatol':2e-13})
            answer.append(float(sign*min(op.fun,sign*values[j,i])))
    return answer


def shoot(u0,T,k):
    reference=u0.copy(); direction=field(reference,k)
    last={}
    def fun(v):
        sol=flow(v[:3],v[3],k); end=sol.y[:3,-1]
        residual=np.r_[end-v[:3],np.dot(v[:3]-reference,direction)]
        J=np.zeros((4,4)); J[:3,:3]=sol.y[3:,-1].reshape(3,3)-np.eye(3)
        J[:3,3]=field(end,k); J[3,:3]=direction
        last.update(sol=sol,residual=residual)
        return residual,J
    fit=root(fun,np.r_[u0,T],jac=True,tol=1e-10)
    residual,_=fun(fit.x)
    assert np.max(abs(residual))<1e-9 and fit.x[3]>0
    return fit.x,last['sol'],float(np.max(abs(residual)))


def main():
    tags=['lower-n40','lower-n80','upper-n80']
    all_data={}; checks=[]; summary={}
    for tag in tags:
        data=np.genfromtxt(OUT/f'branch-{tag}.csv',delimiter=',',names=True)
        all_data[tag]=data
        maxclosure=maxneutral=maxmuerror=0.
        nontrivial=[]; shooting=[]
        for i,row in enumerate(data):
            k=row['kappa']; T=row['period']; u0=np.array([row['x0'],row['y0'],row['z0']])
            sol=flow(u0,T,k); neutral,mu=multipliers(sol)
            closure=float(max(abs(sol.y[:3,-1]-u0)))
            bk=np.exp(np.array([row[f'logmu{j}_re']+1j*row[f'logmu{j}_im'] for j in [1,2,3]]))
            muerror=float(max(abs(np.sort_complex(np.r_[neutral,mu])-np.sort_complex(bk))))
            maxclosure=max(maxclosure,closure); maxneutral=max(maxneutral,abs(neutral-1))
            maxmuerror=max(maxmuerror,muerror); nontrivial.append(float(max(abs(mu))))
            checks.append([tag,i,k,closure,float(abs(neutral-1)),float(max(abs(mu))),muerror])
            if i%10==0 and row['xmax']-row['xmin']>.015:
                fit,s,r=shoot(u0,T,k); ext=extrema(s,fit[3])
                halfspan=(ext[1]-ext[0])/2
                shooting.append(dict(kappa=k,period_abs_error=float(abs(fit[3]-T)),
                    halfamplitude_abs_error=float(abs(halfspan-(row['xmax']-row['xmin'])/2)),
                    closure=r))
        assert maxclosure<2e-5 and maxneutral<2e-5
        assert max(nontrivial)<1.
        direction=1 if tag.startswith('lower') else -1
        assert np.min(direction*np.diff(data['kappa']))>0
        summary[tag]=dict(orbits=len(data),kappa_start=float(data['kappa'][0]),kappa_end=float(data['kappa'][-1]),
              collocation_residual_max=float(max(data['residual'])),IVP_closure_max=float(maxclosure),
              neutral_multiplier_error_max=float(maxneutral),BK_IVP_multiplier_discrepancy_max=maxmuerror,
              transverse_multiplier_modulus_max=max(nontrivial),transverse_multiplier_modulus_min=min(nontrivial),
              shooting_checks=shooting,
              min_state=float(min(min(data['xmin']),min(data['ymin']),min(data['zmin']))),
              max_x_halfamplitude=float(max((data['xmax']-data['xmin'])/2)),
              period_range=[float(min(data['period'])),float(max(data['period']))])
        print(tag,json.dumps({k:v for k,v in summary[tag].items() if k!='shooting_checks'}),flush=True)
    # Match both directions at 21 common speeds using independent phase-fixed
    # shooting, starting from the nearest saved collocation orbit on each branch.
    comparisons=[]
    for k in np.linspace(KM+.04,KP-.04,21):
        found=[]
        for tag in ['lower-n80','upper-n80']:
            data=all_data[tag]; row=data[np.argmin(abs(data['kappa']-k))]
            initial=np.array([row['x0'],row['y0'],row['z0']])
            fit,sol,res=shoot(initial,row['period'],k)
            ex=extrema(sol,fit[3]); assert ex[1]-ex[0]>.1
            found.append(np.r_[fit[3],ex])
        comparisons.append([k,*abs(found[0]-found[1])])
    comparisons=np.array(comparisons)
    assert comparisons[:,1:].max()<1e-7
    summary['bidirectional_common_speed_checks']=dict(count=21,
          period_abs_difference_max=float(max(comparisons[:,1])),
          extrema_abs_difference_max=float(comparisons[:,2:].max()))
    with (OUT/'independent-checks.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['run','index','kappa','closure','neutral_error','transverse_modulus','BK_IVP_mu_difference']);w.writerows(checks)
    np.savetxt(OUT/'bidirectional-common-speeds.csv',comparisons,delimiter=',',
       header='kappa,period_difference,xmin_difference,xmax_difference,ymin_difference,ymax_difference,zmin_difference,zmax_difference',comments='')
    summary['environment']=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
         julia='1.12.4',BifurcationKit='0.8.5',method='PALC; orthogonal collocation; degree 4; 40/80 uniform intervals',
         independent_solver='DOP853, rtol=2e-12, atol=2e-14, max_step=T/80')
    summary['source_sha256']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
        ['Codigo/continuacion_bk.jl','Codigo/validar_continuacion.py','Codigo/continuacion-julia/Manifest.toml']}
    (OUT/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    plot(all_data,checks)
    print(json.dumps(summary['bidirectional_common_speed_checks'],indent=2))


def plot(data,checks):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    low=data['lower-n80'];up=data['upper-n80']; co=data['lower-n40']
    for lang in ['en','es']:
        fig,axs=plt.subplots(3,1,figsize=(6.5,7.2),sharex=True)
        en=lang=='en'
        axs[0].plot(low['kappa'],(low['xmax']-low['xmin'])/2,color='#24587a',lw=1.8,
                    label='From lower Hopf, 80 intervals' if en else 'Desde Hopf inferior, 80 intervalos')
        axs[0].plot(up['kappa'][::4],(up['xmax'][::4]-up['xmin'][::4])/2,'o',mfc='none',ms=4,color='#b94b28',
                    label='From upper Hopf, 80 intervals' if en else 'Desde Hopf superior, 80 intervalos')
        axs[0].plot(co['kappa'][::6],(co['xmax'][::6]-co['xmin'][::6])/2,'+',color='#62686d',ms=6,
                    label='40-interval check' if en else 'Control de 40 intervalos')
        axs[0].plot([.1,KM],[0,0],color='black',lw=2)
        axs[0].plot([KM,KP],[0,0],'--',color='black',lw=1)
        axs[0].plot([KP,6.8],[0,0],color='black',lw=2)
        axs[0].scatter([KM,KP],[0,0],c='black',s=24,zorder=5)
        axs[0].set_ylabel(r'$A_x=(x_{\max}-x_{\min})/2$')
        axs[0].legend(fontsize=8,loc='upper right')
        axs[1].plot(low['kappa'],low['period'],color='#24587a',lw=1.7)
        axs[1].scatter([KM,KP],[2*np.pi/np.sqrt(5+3.8*KM),2*np.pi/np.sqrt(5+3.8*KP)],s=24,c='black',zorder=5)
        axs[1].set_ylabel('Period' if en else 'Periodo')
        rows=np.array([r[2:6] for r in checks if r[0]=='lower-n80'],float)
        axs[2].semilogy(rows[:,0],rows[:,3],color='#24587a',lw=1.7)
        axs[2].axhline(1,ls='--',color='#b94b28',lw=1)
        axs[2].set_ylabel('Largest nontrivial\nFloquet modulus' if en else 'Mayor módulo de\nFloquet no trivial')
        axs[2].set_xlabel(r'Adjustment speed $\kappa$' if en else r'Velocidad de ajuste $\kappa$')
        for ax in axs:
            ax.axvline(KM,ls=':',color='#777777',lw=.8);ax.axvline(KP,ls=':',color='#777777',lw=.8)
            ax.grid(alpha=.17);ax.set_xlim(.1,6.8)
        fig.tight_layout()
        directory=ROOT/('figures' if en else 'figures/es')
        fig.savefig(directory/'global-cycle-continuation.pdf',bbox_inches='tight')
        fig.savefig(OUT/f'global-cycle-continuation-{lang}.png',dpi=160,bbox_inches='tight')
        plt.close(fig)


if __name__=='__main__':main()
