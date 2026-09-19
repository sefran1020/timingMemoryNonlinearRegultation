# Local environment: julia --project=Codigo/continuacion-julia Codigo/continuacion_bk.jl [Ntst] [lower|upper]
# Set JULIA_DEPOT_PATH to Codigo/continuacion-julia/depot before launching Julia.
using BifurcationKit, Accessors, LinearAlgebra, Printf, Serialization
const BK = BifurcationKit
BLAS.set_num_threads(1)
const ROOT = dirname(@__DIR__)
const OUT = joinpath(ROOT,"Resultados","continuacion-global")
mkpath(OUT)
const KM = (261-sqrt(52921))/76
const KP = (261+sqrt(52921))/76

function field!(du,u,p,t=0)
    x,y,z=u
    du[1]=x*(12-x-y-10z)
    du[2]=y*(-2+4x-y-z)
    du[3]=p.kappa*(10*(x+8y)/(81+x+8y)-z)
    du
end
function jac!(J,u,p,t=0)
    x,y,z=u; k=p.kappa; sx=810/(81+x+8y)^2
    J[1,1]=12-2x-y-10z; J[1,2]=-x; J[1,3]=-10x
    J[2,1]=4y; J[2,2]=-2+4x-2y-z; J[2,3]=-y
    J[3,1]=k*sx; J[3,2]=8k*sx; J[3,3]=-k
    J
end
function record_po(x,p;kw...)
    orbit=BK.get_periodic_orbit(p.prob,x,p.p)
    (xmin=minimum(orbit[1,:]),xmax=maximum(orbit[1,:]),
     period=BK.getperiod(p.prob,x,p.p))
end

function main()
    ntst=length(ARGS)>=1 ? parse(Int,ARGS[1]) : 40
    side=length(ARGS)>=2 ? ARGS[2] : "lower"
    tag="$(side)-n$(ntst)"
    println("Julia ",VERSION,"; BifurcationKit ",pkgversion(BK),"; ",tag)
    prob=BifurcationProblem(field!,ones(3),(kappa=.2,),(@optic _.kappa);
           J! = jac!,record_from_solution=(x,p;kw...)->(x=x[1],))
    opts=ContinuationPar(p_min=.15,p_max=6.7,ds=.03,dsmax=.1,max_steps=300,
           detect_bifurcation=3,n_inversion=10,max_bisection_steps=40,
           tol_bisection_eigenvalue=1e-11,nev=3,newton_options=NewtonPar(tol=1e-12))
    eq=continuation(prob,PALC(),opts;normC=norminf,verbosity=0)
    println(eq)
    hopfs=findall(pt->pt.type==:hopf,eq.specialpoint)
    @assert length(hopfs)==2
    ih=side=="lower" ? hopfs[1] : hopfs[2]
    direction=side=="lower" ? 1 : -1
    copts=ContinuationPar(ds=direction*.008,dsmax=.04,dsmin=1e-7,
          p_min=KM+1e-6,p_max=KP-1e-6,max_steps=650,
          newton_options=NewtonPar(tol=1e-10,max_iterations=12),
          nev=3,tol_stability=1e-7,detect_bifurcation=3,
          n_inversion=8,max_bisection_steps=30,
          save_sol_every_step=1,save_eig_every_step=1,detect_fold=true)
    method=Collocation(ntst,4;jacobian=BK.DenseAnalyticalInplace())
    started=time()
    branch=continuation(eq,ih,copts,method;
          alg=PALC(),δp=direction*.002,normC=norminf,
          record_from_solution=record_po,verbosity=0,
          eigsolver=BK.FloquetColl())
    println(branch)
    serialize(joinpath(OUT,"branch-$(tag).jls"),branch)
    coll=BK.get_discretization(BK.getprob(branch))
    open(joinpath(OUT,"branch-$(tag).csv"),"w") do io
        println(io,"step,kappa,period,x0,y0,z0,xmin,xmax,ymin,ymax,zmin,zmax,residual,logmu1_re,logmu1_im,logmu2_re,logmu2_im,logmu3_re,logmu3_im")
        for (idx,sv) in enumerate(branch.sol)
            raw=sv.x; k=sv.p
            interp=BK.POInterpolation(coll,raw)
            period=BK.getperiod(coll,raw,nothing)
            states=hcat([interp(t) for t in range(0,period,length=2001)]...)
            residual=maximum(abs,BK.po_residual(coll,raw,(kappa=k,))[1:end-1])
            eig=branch.eig[idx].eigenvals
            row=Any[sv.step,k,period,states[:,1]...]
            for j in 1:3; push!(row,minimum(states[j,:]),maximum(states[j,:])); end
            push!(row,residual)
            for v in eig; push!(row,real(v),imag(v)); end
            println(io,join(row,","))
        end
    end
    open(joinpath(OUT,"run-$(tag).txt"),"w") do io
        println(io,"Julia=",VERSION,"; BifurcationKit=",pkgversion(BK))
        println(io,"Ntst=",ntst,"; degree=4; Newton tolerance=1e-10; PALC; dsmax=0.04; side=",side)
        println(io,"seconds=",time()-started)
        show(io,branch)
    end
    println("Saved ",length(branch.sol)," orbits; elapsed=",time()-started)
end
main()
