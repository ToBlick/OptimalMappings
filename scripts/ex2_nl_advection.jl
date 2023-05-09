using Plots
using LaTeXStrings
using Base.Threads

using LogExpFunctions
using Distances
using WassersteinDictionaries
using GaussianProcesses

using Gridap, Gridap.FESpaces, Gridap.CellData, Gridap.ODEs
using SparseArrays: spzeros

using LinearAlgebra
using LineSearches
using ForwardDiff

using Test, Printf
using Statistics
using Random

# hyperparameters

const ε = 1e-3
const ϵ = 1e-2
const δ = 0.0

# parameters of the PPDE problem

const nₒₛ = 6
const nₒₜ = 10
const d = 2

const μ = 0.0  # strength of non-linearity
const β = 1e-3  # diffusion constant

function w̄(x, θ) # advecting field
    return 0.2 * VectorValue(cos(θ),sin(θ))
end

const σ = sqrt(5e-3)
u(x,t) = exp(-(abs(x[1] - 0.5)^2 + abs(x[2] - 0.5)^2) / 2 / σ^2) 
u(t::Real) = x -> u(x,t)

# finite element spaces

const N = 32
const highorder = 3

domain = (0,1,0,1)
partition = (N,N)
model = CartesianDiscreteModel(domain,partition)

V₂ = FESpace(
  model,
  ReferenceFE(lagrangian,Float64,highorder),
  conformity=:H1,
  dirichlet_tags="boundary"
)
U₂ = TransientTrialFESpace(V₂,u)
U₀ = U₂(0.0)                        # Trial FESpace

Ω = Triangulation(model)
degree = 2*highorder
dΩ = Measure(Ω,degree)

nrm = sum(∫(u(0.0))dΩ)
nu₀(x) = u(0.0)(x) / nrm                # initial condition

# time-stepping

const θ = 0.5   # time-step method

const t0 = 0.0
const tF = 1.0
const dt = 0.05   # time-step

nls = NLSolver(show_trace=false, method=:newton, linesearch=BackTracking())
ode_solver = ThetaMethod(nls,dt,θ)

include("../src/utils.jl")
include("../src/ex2_utils.jl")

# obtain the training set
ω̄_samples = collect(range(0, 2π, length=nₒₛ))
println("Calculating training snapshots...")
uₕ, ω̄, t̄ = snapshots(nu₀, ω̄_samples, V₂, U₂, dΩ)
nₛ = length(uₕ)

# snapshot correlation matrix
C = [ sum(∫((uₕ[i]) ⋅ (uₕ[j]) )dΩ) for i in eachindex(uₕ), j in eachindex(uₕ) ]
evd = eigen(C, sortby = x -> -abs(x) );

# reduced basis

n = get_n(evd.values, ϵ^2)
ϕ = get_ϕ(n, uₕ, evd, V₂, V₂)

# test set
Random.seed!(1234)
ω̄ₜ_samples = rand(nₒₜ) * 2π
println("Calculating test snapshots...")
uₕₜ, ω̄ₜ, t̄ₜ = snapshots(nu₀, ω̄ₜ_samples, V₂, U₂, dΩ)
println("Solving RB problem...")
uₕₜ_rb = snapshots_rb(nu₀, ω̄ₜ_samples, ϕ, V₂, U₂, dΩ)

L2(e) = sqrt(sum( ∫( e*e )*dΩ ))
H1(e) = sqrt(sum( ∫( e*e + ∇(e)⋅∇(e) )*dΩ ))

# errors
ΔL2_rb = [ L2(uₕₜ_rb[i]-uₕₜ[i]) / L2(uₕₜ[i]) for i in eachindex(uₕₜ) ]
ΔH1_rb = [ H1(uₕₜ_rb[i]-uₕₜ[i]) / H1(uₕₜ[i]) for i in eachindex(uₕₜ) ]

### Transport
println("Calculating densities...")
N_fine = highorder*N
partition_fine = (N_fine,N_fine)
model_fine = CartesianDiscreteModel(domain,partition_fine)

V₁ = FESpace(
  model_fine,
  ReferenceFE(lagrangian,Float64,1),
  conformity=:H1
)

Ψ = FESpace(
  model,
  ReferenceFE(lagrangian,Float64,highorder),
  conformity=:H1
)

c = WassersteinDictionaries.get_cost_matrix_separated(N_fine+1, d, a = [domain[1] domain[3]], b = [domain[2] domain[4]])
k = WassersteinDictionaries.get_gibbs_matrix(c, ε)

# ρ
ūₕ = [ reshape( abs.(get_free_dof_values( interpolate_everywhere(Interpolable(_uₕ), V₁) )),N_fine+1,N_fine+1) for _uₕ in uₕ]
for _ūₕ in ūₕ
    _ūₕ ./= sum(_ūₕ) * (1 / N_fine)^2
end
log_ūₕ = [ safe_log.(_ūₕ) for _ūₕ in ūₕ ]

# calculate reference density
SPB = SinkhornParameters(256, ε)
SPB.tol = 1e-12
SPB.debias = true
SPB.averaged_updates = false
SPB.update_potentials = false
MC = MatrixCache(N_fine+1)

Random.seed!(2345)
nᵦ = 25
ūₕᵦ = ūₕ[rand(1:nₛ,nᵦ)]

ūₕ_ref = sinkhorn_barycenter_sep([ 1/nᵦ for _ in ūₕᵦ], ūₕᵦ, k, SPB, MC)
log_ūₕ_ref = safe_log.(ūₕ_ref)
uₕ_ref = interpolate_everywhere( Interpolable( FEFunction(V₁, vec(ūₕ_ref))), Ψ)

# transport potentials
SP = SinkhornParameters(128, ε)
SP.tol = 1e-12
SP.debias = false
SP.averaged_updates = false
SP.update_potentials = true

println("Calculating transport potentials...")
ψ̄ᶜ = get_ψ̄ᶜ(ūₕ, ūₕ_ref, SP, MC)
println("Enforcing boundary conditions...")
ψᶜ = boundary_projection(ψ̄ᶜ, δ, N_fine, model, degree, V₁, Ψ, dΩ)

# Monge embedding correlation matrix
Cψ = [ sum(∫(∇(ψᶜᵢ)⋅∇(ψᶜⱼ)*uₕ_ref)dΩ) for ψᶜᵢ in ψᶜ, ψᶜⱼ in ψᶜ ]
evdψ = eigen(Cψ, sortby = x -> -abs(x))
m = get_n(evdψ.values, ϵ)

# transport modes
ξᶜ = get_ϕ(m, ψᶜ, evdψ, Ψ, Ψ)

println("Plotting transport modes...")
# plot transport modes
x_cord = 0:0.01:1
for i in 1:m
    surface(x_cord,x_cord, (x,y) -> ξᶜ[i](Point(x,y)) - ξᶜ[i](Point(0.5,0.5)),
            cbar = false, xlabel = L"x", ylabel = L"y", zlabel = L"\xi^c_{%$i}",  size = (400,400), camera = (30,30),
            legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, zguidefontsize=12, dpi=100)
    savefig("../figs/xi$i.png")
end

# get the reduced representation of transport potentials
λ = [ [ sum(∫(∇(_ψᶜ)⋅∇(_ξᶜ)*uₕ_ref)dΩ) for _ξᶜ in ξᶜ ] for _ψᶜ in ψᶜ ]
ψᶜᵣ = [ FEFunction(Ψ, _λ' * get_free_dof_values.(ξᶜ)) for _λ in λ ]

# errors
L2_ψ = [ sum(∫(∇(ψᶜ[i] - ψᶜᵣ[i])⋅∇(ψᶜ[i] - ψᶜᵣ[i])*uₕ_ref)dΩ) for i in eachindex(ψᶜ) ] #/ sum(∫(∇(ψᶜ[i])⋅∇(ψᶜ[i])*uₕ_ref)dΩ) 

println("Calculating reference basis...")
# map to reference domain
Tuₕ = copy(uₕ)
for i in eachindex(uₕ) 
    Tuₕ[i] = pushfwd(uₕ[i], ψᶜᵣ[i], V₂, domain, dΩ)
end

### plot some snapshot crossections before and after registration
cpal = palette(:thermal, 7);
plt1 = plot()
_i = 1
idxs = rand(1:nₛ,5)
for i in idxs
    plot!(0:0.001:1, x -> Tuₕ[i](Point(x,0.5)), legend = false,
    linewidth = 2, color = cpal[_i], xlabel = L"u", ylabel = L"u \circ \Phi^{-1}",
    legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, dpi=100)
    global _i += 1
end
plt2 = plot()
_i = 1
for i in idxs
    plot!(0:0.001:1, x -> uₕ[i](Point(x,0.5)), legend = false,
    linewidth = 2, color = cpal[_i], xlabel = L"x", ylabel = L"u",
    legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, dpi=100)
    global _i += 1
end
plot(plt1,plt2)
savefig("../figs/u_registered.png")


# mapped snapshots correlation matrix
CT = [ sum(∫( (uₕᵢ) ⋅ (uₕⱼ))dΩ) for uₕᵢ in Tuₕ, uₕⱼ in Tuₕ ]
evdT = eigen(CT, sortby = x -> -abs(x))
nₘ = get_n(evdT.values, ϵ)

# plot ev decay
cpal = palette(:thermal, 4);
plot(abs.(evd.values[1:nₛ]) ./ evd.values[1], yaxis = :log, minorgrid = true, xaxis = :log,
    yticks = 10.0 .^ (-16:2:0), ylim = (1e-16,2), linewidth=2, marker = :circle, 
    xlabel = L"n", ylabel = L"\lambda_n / \lambda_1", label = L"u", color = cpal[1],
    legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, dpi=100)
plot!(abs.(evdT.values[1:nₛ]) ./ evdT.values[1], linewidth=2, marker = :square, 
        markersize = 3, label = L"u \circ \Phi^{-1}", color = cpal[2], dpi=100)
plot!(abs.(evdψ.values[1:nₛ]) ./ evdψ.values[1], linewidth=2, marker = :diamond, 
        label = L"\nabla \psi^c", color = cpal[3], dpi=100)
#savefig("../figs/evds_log.png")
savefig("figs/evds_log.png")

# reference reduced basis
ϕσ = get_ϕ(nₘ, Tuₕ, evdT, V₂, V₂)

# gaussian process approx
_μ = zeros(2, length(ω̄))
for i in eachindex(ω̄)
    _μ[2,i] = ω̄[i]
    _μ[1,i] = t̄[i]
end

gp = get_gp(_μ, λ, m)

# plot gp
for i in 1:m
    surface(gp[i]; legend = false, xlabel = L"t", ylabel = L"θ", zlabel = L"w_{%$i}",
            xlim = (t0,tF), ylim = (0,2π), camera = (30, 45), size = (400,400),
            legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, dpi=100)
    scatter!(t̄, ω̄, predict_y(gp[1], _μ)[1], label = false, color = :white, markersize = 2, aspectratio=1, dpi=100)
    savefig("../figs/w$i.png")
end

println("Solving registered RB problem...")
uₕₜ_trb, uₕₜ_trb_ref = snapshots_trb(nu₀, ω̄ₜ_samples, ϕσ, V₂, U₂, dΩ, domain, ξᶜ, gp, Ψ, V₁, N_fine, log_ūₕ_ref, MC)

# errors
ΔL2_trb = [ L2(uₕₜ_trb[i]-uₕₜ[i]) / L2(uₕₜ[i]) for i in eachindex(uₕₜ_trb) ]
ΔH1_trb = [ H1(uₕₜ_trb[i]-uₕₜ[i]) / H1(uₕₜ[i]) for i in eachindex(uₕₜ_trb) ]

M_ΔL2_rb = reshape(ΔL2_rb, :, nₒₜ)'
M_ΔL2_trb = reshape(ΔL2_trb, :, nₒₜ)'
M_ΔH1_rb = reshape(ΔH1_rb, :, nₒₜ)'
M_ΔH1_trb = reshape(ΔH1_trb, :, nₒₜ)'

# plot the L2 error over time
cpal = palette(:thermal, 4)
plot(t0:dt:tF, Statistics.mean(M_ΔL2_trb, dims = 1)', minorgrid = true, yaxis = :log,
    ribbon = ( Statistics.mean(M_ΔL2_trb, dims = 1)' - minimum(M_ΔL2_trb, dims = 1)',
                - Statistics.mean(M_ΔL2_trb, dims = 1)' + maximum(M_ΔL2_trb, dims = 1)'),
    linewidth = 2, color = cpal[1], label = L"u_{\mathrm{trb}}" )
plot!(t0:dt:tF, Statistics.mean(M_ΔL2_rb, dims = 1)',
    ribbon = ( Statistics.mean(M_ΔL2_rb, dims = 1)' - minimum(M_ΔL2_rb, dims = 1)',
               - Statistics.mean(M_ΔL2_rb, dims = 1)' + maximum(M_ΔL2_rb, dims = 1)'),
    linewidth = 2, color = cpal[3], xlabel = L"t", ylabel = L"\Vert u - u_{\{\mathrm{rb,trb}\}} \Vert_{L^2} / \Vert u \Vert_{L^2}" , 
    legend = :topleft, label = L"u_{\mathrm{rb}}",
    legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12
)


println("ε = $ε, ϵ = $ϵ")

@printf "registered (n = %.0f, m = %.0f) \t L2 error at t = %.1f avg.: %.2e ± %.2e \t max.: %.2e \n" nₘ m tF Statistics.mean(M_ΔL2_trb[:,end]) Statistics.std(M_ΔL2_trb[:,end]; corrected=false) maximum(M_ΔL2_trb[:,end])
@printf "unregistered (n = %.0f, m = %.0f) \t L2 error at t = %.1f avg.: %.2e ± %.2e \t max.: %.2e \n" n 0 tF Statistics.mean(M_ΔL2_rb[:,end]) Statistics.std(M_ΔL2_rb[:,end]; corrected=false) maximum(M_ΔL2_rb[:,end])

@printf "registered (n = %.0f, m = %.0f) \t H1 error at t = %.1f avg.: %.2e ± %.2e \t max.: %.2e \n" nₘ m tF Statistics.mean(M_ΔH1_trb[:,end]) Statistics.std(M_ΔH1_trb[:,end]; corrected=false) maximum(M_ΔH1_trb[:,end])
@printf "unregistered (n = %.0f, m = %.0f) \t H1 error at t = %.1f avg.: %.2e ± %.2e \t max.: %.2e" n 0 tF Statistics.mean(M_ΔH1_rb[:,end]) Statistics.std(M_ΔH1_rb[:,end]; corrected=false) maximum(M_ΔH1_rb[:,end])