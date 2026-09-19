# Continuación de codimensión dos para M2 con objetivo de Hill (AD-18 a AD-21, CN-4).
# Entorno local: julia --project=Codigo/continuacion-julia Codigo/codim2_hill_bk.jl [n] [h]
# Fijar JULIA_DEPOT_PATH=Codigo/continuacion-julia/depot antes de lanzar Julia.
# Modalidad C: detección numérica de puntos especiales; no sustituye las pruebas
# exactas de la sección de codimensión dos del informe.
using BifurcationKit, Accessors, LinearAlgebra, Printf, Serialization
const BK = BifurcationKit
BLAS.set_num_threads(1)
const ROOT = dirname(@__DIR__)
const OUT = joinpath(ROOT, "Resultados", "codim2-hill")
mkpath(OUT)

const NH = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 2.0
const HH = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 81.0

g(s, Z) = Z * s^NH / (HH^NH + s^NH)
gp(s, Z) = Z * NH * HH^NH * s^(NH - 1) / (HH^NH + s^NH)^2

# Tasas del ejemplo racional: r=K=12, a=d=e=1, b=10, c=4, m=2, theta=1, phi=8.
function field!(du, u, p, t = 0)
    x, y, z = u
    s = x + 8y
    du[1] = x * (12 - x - y - 10z)
    du[2] = y * (-2 + 4x - y - z)
    du[3] = p.kappa * (g(s, p.Z) - z)
    du
end
field(u, p) = field!(similar(u), u, p)

function jac!(J, u, p, t = 0)
    x, y, z = u
    k = p.kappa
    sx = gp(x + 8y, p.Z)
    J[1, 1] = 12 - 2x - y - 10z; J[1, 2] = -x; J[1, 3] = -10x
    J[2, 1] = 4y; J[2, 2] = -2 + 4x - 2y - z; J[2, 3] = -y
    J[3, 1] = k * sx; J[3, 2] = 8k * sx; J[3, 3] = -k
    J
end
jac(u, p) = jac!(zeros(eltype(u), 3, 3), u, p)

"Equilibrio interior por la reducción escalar F(z)=g(G(z))-z (bisección)."
function equilibrio(Z)
    X0, Y0, A, B0, zeta = 14 / 5, 46 / 5, -9 / 5, 41 / 5, 46 / 41
    F(z) = g((X0 + A * z) + 8 * (Y0 - B0 * z), Z) - z
    lo, hi = 0.0, zeta
    @assert F(lo) > 0 && F(hi) < 0
    for _ in 1:200
        mid = (lo + hi) / 2
        F(mid) > 0 ? (lo = mid) : (hi = mid)
    end
    z = (lo + hi) / 2
    [X0 + A * z, Y0 - B0 * z, z]
end

function main()
    println("Julia ", VERSION, "; BifurcationKit ", pkgversion(BK), "; Hill n=", NH, " h=", HH)
    Z0 = 20.0
    prob = BifurcationProblem(field!, equilibrio(Z0), (Z = Z0, kappa = 0.05), (@optic _.kappa);
        J = jac, record_from_solution = (x, p; kw...) -> (x = x[1], y = x[2], z = x[3]))
    opts = ContinuationPar(p_min = 1e-3, p_max = 30.0, ds = 0.01, dsmax = 0.05, max_steps = 2000,
        detect_bifurcation = 3, n_inversion = 10, max_bisection_steps = 40,
        tol_stability = 1e-10, nev = 3, newton_options = NewtonPar(tol = 1e-12))
    eq = continuation(prob, PALC(), opts; normC = norminf, verbosity = 0)
    println(eq)
    hopfs = findall(pt -> pt.type == :hopf, eq.specialpoint)
    println("Hopf en kappa: ", [eq.specialpoint[i].param for i in hopfs])
    resumen = Dict{String,Any}("n" => NH, "h" => HH, "Z0" => Z0)
    curvas = Any[]
    for (j, ih) in enumerate(hopfs)
        nf = get_normal_form(eq, ih; nev = 3)
        @printf("Hopf %d: kappa=%.10f  l1(BK)=%.8f\n", j, eq.specialpoint[ih].param, real(nf.nf.b))
        for dir in (1, -1)
            copts = ContinuationPar(opts; p_min = 0.3, p_max = 118.0, ds = dir * 0.01, dsmax = 0.05,
                max_steps = 4000, detect_bifurcation = 1, n_inversion = 6)
            hc = continuation(eq, ih, (@optic _.Z), copts;
                start_with_eigen = true, detect_codim2_bifurcation = 2,
                update_minaug_every_step = 1, bdlinsolver = MatrixBLS(),
                normC = norminf, verbosity = 0)
            println("Curva de Hopf ", j, " sentido ", dir, ": ", length(hc), " puntos")
            j == 1 && dir == 1 && println("   campos registrados: ", keys(hc.branch[1]))
            for sp in hc.specialpoint
                sp.type in (:endpoint,) && continue
                println("   ", sp.type, "  Z=", sp.param, "  ", sp.printsol)
            end
            push!(curvas, (j = j, dir = dir, branch = hc))
        end
    end
    # Puntos especiales de codimensión dos detectados en todas las curvas
    especiales = Any[]
    for c in curvas, sp in c.branch.specialpoint
        sp.type in (:endpoint, :hopf) && continue
        fila = Dict{String,Any}("tipo" => String(sp.type), "Z" => sp.param,
            "kappa" => sp.printsol[:kappa], "curva" => c.j, "sentido" => c.dir)
        if sp.type == :gh
            try
                nfb = get_normal_form(c.branch, findfirst(==(sp), c.branch.specialpoint); nev = 3)
                fila["l2"] = real(nfb.nf.l2)
                fila["omega"] = real(nfb.nf.ω)
            catch err
                fila["l2_error"] = sprint(showerror, err)
            end
        end
        push!(especiales, fila)
    end
    resumen["especiales"] = especiales
    tag = @sprintf("n%g-h%g", NH, HH)
    open(joinpath(OUT, "curvas-hopf-$(tag).csv"), "w") do io
        println(io, "curva,sentido,Z,kappa,omega2,l1")
        for c in curvas, (k, pt) in enumerate(c.branch.branch)
            l1 = hasproperty(pt, :l1) ? real(pt.l1) : NaN
            println(io, join((c.j, c.dir, pt.Z, pt.kappa, hasproperty(pt, :ω) ? pt.ω : NaN, l1), ","))
        end
    end
    open(joinpath(OUT, "especiales-$(tag).txt"), "w") do io
        println(io, "Julia=", VERSION, "; BifurcationKit=", pkgversion(BK), "; n=", NH, "; h=", HH)
        for f in especiales
            println(io, f)
        end
    end
    println(especiales)
end

main()
