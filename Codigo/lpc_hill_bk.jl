# Ciclos en kappa a Z fijo con objetivo de Hill (n=2, h=81), desde el Hopf superior hacia
# kappa decreciente: atraviesan la ventana y detectan el pliegue de ciclos (LPC) que
# acota la histéresis por debajo de kappa_-. Ver también Codigo/bautin_l2.py.
# Control independiente del signo de l2 en el punto GH (Codigo/bautin_l2.py):
#   l2 > 0  => el pliegue de ciclos (LPC) cercano aparece del lado supercrítico (Z < Z_GH),
#              dentro de la ventana (kappa > kappa_-);
#   l2 < 0  => aparece del lado subcrítico (Z > Z_GH), fuera de la ventana (kappa < kappa_-).
# Uso: julia --project=Codigo/continuacion-julia Codigo/lpc_hill_bk.jl Z
using BifurcationKit, Accessors, LinearAlgebra, Printf
const BK = BifurcationKit
BLAS.set_num_threads(1)
const ROOT = dirname(@__DIR__)
const OUT = joinpath(ROOT, "Resultados", "codim2-hill")
mkpath(OUT)
const ZF = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 29.5
const NTST = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 60

g(s, Z) = Z * s^2 / (81.0^2 + s^2)
gp(s, Z) = Z * 2 * 81.0^2 * s / (81.0^2 + s^2)^2
function field!(du, u, p, t = 0)
    x, y, z = u
    du[1] = x * (12 - x - y - 10z)
    du[2] = y * (-2 + 4x - y - z)
    du[3] = p.kappa * (g(x + 8y, p.Z) - z)
    du
end
function jac!(J, u, p, t = 0)
    x, y, z = u; k = p.kappa; sx = gp(x + 8y, p.Z)
    J[1, 1] = 12 - 2x - y - 10z; J[1, 2] = -x; J[1, 3] = -10x
    J[2, 1] = 4y; J[2, 2] = -2 + 4x - 2y - z; J[2, 3] = -y
    J[3, 1] = k * sx; J[3, 2] = 8k * sx; J[3, 3] = -k
    J
end
function equilibrio(Z)
    X0, Y0, A, B0, zeta = 14 / 5, 46 / 5, -9 / 5, 41 / 5, 46 / 41
    F(z) = g((X0 + A * z) + 8 * (Y0 - B0 * z), Z) - z
    lo, hi = 0.0, zeta
    for _ in 1:200
        mid = (lo + hi) / 2
        F(mid) > 0 ? (lo = mid) : (hi = mid)
    end
    z = (lo + hi) / 2
    [X0 + A * z, Y0 - B0 * z, z]
end
record_po(x, p; kw...) = begin
    orbit = BK.get_periodic_orbit(p.prob, x, p.p)
    (xmin = minimum(orbit[1, :]), xmax = maximum(orbit[1, :]), period = BK.getperiod(p.prob, x, p.p))
end

function main()
    prob = BifurcationProblem(field!, equilibrio(ZF), (Z = ZF, kappa = 0.2), (@optic _.kappa);
        J! = jac!, record_from_solution = (x, p; kw...) -> (x = x[1],))
    opts = ContinuationPar(p_min = 0.05, p_max = 20.0, ds = 0.01, dsmax = 0.02, max_steps = 400,
        detect_bifurcation = 3, n_inversion = 10, max_bisection_steps = 40,
        tol_bisection_eigenvalue = 1e-12, nev = 3, newton_options = NewtonPar(tol = 1e-12))
    eq = continuation(prob, PALC(), opts; normC = norminf, verbosity = 0)
    hopfs = findall(pt -> pt.type == :hopf, eq.specialpoint)
    km = eq.specialpoint[hopfs[1]].param
    ih = hopfs[end]
    @printf("Z=%.4f  kappa_-=%.10f  kappa_+=%.10f\n", ZF, km, eq.specialpoint[ih].param)
    copts = ContinuationPar(ds = -0.002, dsmax = 0.02, dsmin = 1e-9, p_min = 0.05, p_max = 20.0,
        max_steps = 700, newton_options = NewtonPar(tol = 1e-10, max_iterations = 15),
        nev = 3, tol_stability = 1e-7, detect_bifurcation = 3, n_inversion = 8,
        max_bisection_steps = 30, detect_fold = true)
    method = Collocation(NTST, 4; jacobian = BK.DenseAnalyticalInplace())
    br = continuation(eq, ih, copts, method; alg = PALC(), δp = -0.002, normC = norminf,
        record_from_solution = record_po, verbosity = 0, eigsolver = BK.FloquetColl())
    println(br)
    open(joinpath(OUT, @sprintf("ciclos-Z%.2f.csv", ZF)), "w") do io
        println(io, "kappa,xmin,xmax,period,stable")
        for pt in br.branch
            println(io, join((pt.param, pt.xmin, pt.xmax, pt.period, pt.stable), ","))
        end
    end
    # primer giro de la rama en kappa (pliegue de ciclos) y resumen
    ks = [pt.param for pt in br.branch]
    igiro = findfirst(i -> (ks[i] - ks[i-1]) * (ks[i+1] - ks[i]) < 0, 2:length(ks)-1)
    klpc = igiro === nothing ? NaN : ks[igiro + 1]
    @printf("RESUMEN Z=%.4f kappa_-=%.10f kappa_+=%.10f kappa_LPC=%.10f\n", ZF, km,
        eq.specialpoint[ih].param, klpc)
    open(joinpath(OUT, "lpc-resumen.csv"), "a") do io
        println(io, join((ZF, km, eq.specialpoint[ih].param, klpc, NTST), ","))
    end
end
main()
