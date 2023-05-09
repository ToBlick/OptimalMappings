using Plots
using LaTeXStrings
using Base.Threads

using LogExpFunctions
using Distances
using WassersteinDictionaries
using GaussianProcesses

using Gridap, Gridap.FESpaces, Gridap.CellData
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
const δ = 1e-9

# parameters of the PPDE problem

const nₛ = 100
const nₜ = 50
const d = 2

# parameter space
const μ_min = -0.35
const μ_max = 0.35

# initial condition
const σ = sqrt(1e-3)
f(x, x0) = exp(-((x[1] - x0[1])^2 + (x[2] - x0[2])^2) / 2 / σ^2) / (2π*σ^2)

# finite element spaces
const N = 64
const highorder = 3

domain = (0,1,0,1)
partition = (N,N)
model = CartesianDiscreteModel(domain,partition)
u(x) = 0.0

V₂ = FESpace(
  model,
  ReferenceFE(lagrangian,Float64,highorder),
  conformity=:H1,
  dirichlet_tags="boundary"
)
U₂ = TrialFESpace(V₂,u)

Ω = Triangulation(model)
degree = 2*highorder
dΩ = Measure(Ω,degree)

include("../src/utils.jl")
include("../src/ex1_utils.jl")

# obtain the training set
Random.seed!(1234)
μ̄ = trainparameterset(nₛ, μ_max, μ_min)
println("Calculating training snapshots...")
uₕ = snapshots(f, μ̄, V₂, U₂, dΩ)

# snapshot correlation matrix
C = [ sum(∫((uₕ[i]) ⋅ (uₕ[j]) )dΩ) for i in eachindex(uₕ), j in eachindex(uₕ) ]
evd = eigen(C, sortby = x -> -abs(x) );

# reduced basis
n = get_n(evd.values, ϵ)
ϕ = get_ϕ(n, uₕ, evd, V₂, U₂)

# test set
Random.seed!(4932)
μ̄ₜ = testparameterset(nₜ, μ_max, μ_min)
println("Calculating test snapshots...")
uₕₜ = snapshots(f, μ̄ₜ, V₂, U₂, dΩ)
println("Solving RB problem...")
uₕₜ_rb = snapshots_rb(f, μ̄ₜ, ϕ, V₂, U₂, dΩ)

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
  ReferenceFE(lagrangian,Float64,3),
  conformity=:H1
)

c = WassersteinDictionaries.get_cost_matrix_separated(N_fine+1, d, a = [domain[1] domain[3]], b = [domain[2] domain[4]])
k = WassersteinDictionaries.get_gibbs_matrix(c, ε)

# ρ
ūₕ = [ reshape( abs.(get_free_dof_values( interpolate_everywhere(Interpolable((_uₕ)⋅(_uₕ)), V₁) )),N_fine+1,N_fine+1) for _uₕ in uₕ]
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

Random.seed!(2424)
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
L2_ψ = [ sum(∫(∇(ψᶜ[i] - ψᶜᵣ[i])⋅∇(ψᶜ[i] - ψᶜᵣ[i])*uₕ_ref)dΩ) / sum(∫(∇(ψᶜ[i])⋅∇(ψᶜ[i])*uₕ_ref)dΩ) for i in eachindex(ψᶜ) ]

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
plot!(abs.(evdT.values[1:nₛ]) ./ evdT.values[1], linewidth=2, marker = :square, markersize = 3, 
    label = L"u \circ \Phi^{-1}", color = cpal[2], dpi=100)
plot!(abs.(evdψ.values[1:nₛ]) ./ evdψ.values[1], linewidth=2, marker = :diamond, 
    label = L"\nabla \psi^c", color = cpal[3], dpi=100)
savefig("../figs/evds_log.png")

# reference reduced basis
ϕσ = get_ϕ(nₘ, Tuₕ, evdT, V₂, V₂)

# gaussian process approx
_μ = [ μ̄[i][k] for k in 1:d, i in eachindex(μ̄)]
gp = get_gp(_μ, λ, m)

# plot gp
μ̄₁ = [μ̄[i][1] for i in eachindex(μ̄)]
μ̄₂ = [μ̄[i][2] for i in eachindex(μ̄)]
_μ = [ μ̄[i][k] for k in 1:d, i in eachindex(μ̄)]
for i in 1:m
    surface(gp[i]; legend = false, xlabel = L"\mu_1", ylabel = L"\mu_2", zlabel = L"w_{%$i}",
    xlim = (μ_min,μ_max), ylim = (μ_min,μ_max), camera = (30, 45), size = (400,400),
    legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, dpi=100)
    scatter!(μ̄₁, μ̄₂, predict_y(gp[i], _μ)[1], label = false, color = :white, markersize = 2, aspectratio=1, dpi=100)
    savefig("../figs/w$i.png")
end

println("Solving registered RB problem...")
uₕₜ_trb, uₕₜ_trb_ref = snapshots_trb(f, μ̄ₜ, ϕσ, V₂, U₂, dΩ, domain, ξᶜ, gp, Ψ, V₁, N_fine, log_ūₕ_ref, MC)

# errors
ΔL2_trb = [ L2(uₕₜ_trb[i]-uₕₜ[i]) / L2(uₕₜ[i]) for i in eachindex(uₕₜ_trb) ]
ΔH1_trb = [ H1(uₕₜ_trb[i]-uₕₜ[i]) / H1(uₕₜ[i]) for i in eachindex(uₕₜ_trb) ]

println("ε = $ε, ϵ = $ϵ")

@printf "registered (n = %.0f, m = %.0f) \t L2 error avg.: %.2e ± %.2e \t max.: %.2e \n" nₘ m Statistics.mean(ΔL2_trb) Statistics.std(ΔL2_trb; corrected=false) maximum(ΔL2_trb)
@printf "unregistered (n = %.0f, m = %.0f) \t L2 error avg.: %.2e ± %.2e \t max.: %.2e \n" n 0 Statistics.mean(ΔL2_rb) Statistics.std(ΔL2_rb; corrected=false) maximum(ΔL2_rb)

@printf "registered (n = %.0f, m = %.0f) \t H1 error avg.: %.2e ± %.2e \t max.: %.2e \n" nₘ m Statistics.mean(ΔH1_trb) Statistics.std(ΔH1_trb; corrected=false) maximum(ΔH1_trb)
@printf "unregistered (n = %.0f, m = %.0f) \t H1 error avg.: %.2e ± %.2e \t max.: %.2e \n" n 0 Statistics.mean(ΔH1_rb) Statistics.std(ΔH1_rb; corrected=false) maximum(ΔH1_rb)

### print the worst case (H1 error) reconstruction

_i = argmax(ΔH1_trb)
μ = μ̄ₜ[_i]
cpal = palette(:thermal, 4)
plt1 = plot()
plot!(0:0.001:1, x -> uₕₜ[_i](Point([x,0.5+μ[2]])), label = L"u", xlabel = L"x",
linewidth = 2, color = cpal[1],
legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, dpi=100)
plot!(0:0.001:1, x -> uₕₜ_rb[_i](Point([x,0.5+μ[2]])), label = L"u_{\mathrm{rb}}",
linewidth = 2, color = cpal[2],
legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, dpi=100)
plot!(0:0.001:1, x -> uₕₜ_trb[_i](Point([x,0.5+μ[2]])), label = L"u_{\mathrm{trb}}",
linewidth = 2, color = cpal[3],
legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, dpi=100)
plt2 = plot()
plot!(0:0.001:1, x -> uₕₜ[_i](Point([0.5+μ[1],x])), label = L"u", xlabel = L"y",
linewidth = 2, color = cpal[1],
legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, dpi=100)
plot!(0:0.001:1, x -> uₕₜ_rb[_i](Point([0.5+μ[1],x])), label = L"u_{\mathrm{rb}}",
linewidth = 2, color = cpal[2],
legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, dpi=100)
plot!(0:0.001:1, x -> uₕₜ_trb[_i](Point([0.5+μ[1],x])), label = L"u_{\mathrm{trb}}",
linewidth = 2, color = cpal[3],
legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, dpi=100)
plot(plt1, plt2)
savefig("../figs/crosssecs.png")



