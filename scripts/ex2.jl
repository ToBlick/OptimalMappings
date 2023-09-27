using OptimalMappings
using Printf
using LaTeXStrings
using Random
using BenchmarkTools
using WassersteinDictionaries
using Gridap, Gridap.FESpaces, Gridap.CellData
using Gridap.CellData: get_cell_quadrature, get_node_coordinates
using LineSearches
using Revise
using Plots
using GaussianProcesses
using Statistics

# hyperparameters
const ε = 1e-2
const τ = 1e-3
const τ_eim = τ

const δ⁻¹ = 1e9
const κ = 1 / sqrt(ε)
const debias = false

# parameters of the PPDE problem

const nₒₛ = 10
const nₒₜ = 10
const d = 2

const γ = 0.01  # strength of non-linearity
const β = 1e-3  # diffusion constant

function w̄(x, α) # advecting field
    return 0.2 * VectorValue(cos(α), sin(α))
end

P = NonlinearAdvectionProblem(β, γ, w̄)

const σ = sqrt(5e-3)
u(x, t) = exp(-(abs(x[1] - 0.5)^2 + abs(x[2] - 0.5)^2) / 2 / σ^2)
u(t::Real) = x -> u(x, t)

# finite element spaces

const N = 32
const highorder = 3

domain = (0, 1, 0, 1)
partition = (N, N)
model = CartesianDiscreteModel(domain, partition)

V₂ = FESpace(
    model,
    ReferenceFE(lagrangian, Float64, highorder),
    conformity=:H1,
    dirichlet_tags="boundary"
)
U₂ = TransientTrialFESpace(V₂, u)
U₀ = U₂(0.0)                        # Trial FESpace

Ω = Triangulation(model)
degree = 2 * highorder
dΩ = Measure(Ω, degree)

W = FESpace(model, ReferenceFE(lagrangian, Float64, highorder), conformity=:H1);
W_matrix = TestFESpace(model, ReferenceFE(lagrangian, TensorValue{2,2,Float64,4}, highorder); conformity=:H1);
W_vector = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, highorder); conformity=:H1);

nrm = sum(∫(u(0.0))dΩ)
nu₀(x) = u(0.0)(x) / nrm                # initial condition

# time-stepping

const θ = 0.5   # time-step method

const t0 = 0.0
const tF_train = 0.8
const tF_test = 1.0
const dt = 0.05   # time-step

nls = NLSolver(show_trace=false, method=:newton, linesearch=BackTracking())
ode_solver = ThetaMethod(nls, dt, θ)

# obtain the training set
ω_train = collect(range(0, 2π, length=nₒₛ))
t_train = collect(t0:dt:tF_train)

println("Calculating training snapshots...")
u_train, ω_train_all, t_train_all = snapshots(nu₀, ω_train, V₂, U₂, dΩ, nls, θ, t0, dt, tF_train, P)
nₛ = length(u_train)
ζ, evd_u = pod(u_train, V₂, U₀, dΩ, τ)

println("Performing the offline EIM calculations...")
ā = vec([get_ā(w̄, ω, W_vector) for ω in ω_train, t in t_train])
Ξ_ā, evd_ā = pod(ā, W_vector, W_vector, dΩ, τ_eim)
eim_ā = EmpiricalInterpolation(Ξ_ā, (ϕ, X) -> form_ā(ϕ, X, dΩ), ζ, W_vector)
Q_ā = get_Q(eim_ā)
eim_āu = EmpiricalInterpolation(Ξ_ā, (ϕ, X) -> form_āu(ϕ, X, dΩ), ζ, W_vector)
Q_āu = get_Q(eim_āu)

# test set
Random.seed!(1234)
ω_test = rand(nₒₜ) * 2π
println("Calculating test snapshots...")
u_test, ω_test_all, t_test_all = snapshots(nu₀, ω_test, V₂, U₂, dΩ, nls, θ, t0, dt, tF_test, P);
println("Solving RB problem...")
u_rb_eim = snapshots(nu₀, ω_test, ζ, V₂, U₂, dΩ, nls, θ, t0, dt, tF_test, eim_ā, eim_āu, P);

### Transport
println("Calculating densities...")
N_fine = highorder * N
partition_fine = (N_fine, N_fine)
model_fine = CartesianDiscreteModel(domain, partition_fine)
V_fine = FESpace(model_fine, ReferenceFE(lagrangian, Float64, 1), conformity=:H1)
Ψ = FESpace(model, ReferenceFE(lagrangian, Float64, highorder), conformity=:H1)

c = WassersteinDictionaries.get_cost_matrix_separated(N_fine + 1, d, a=[domain[1] domain[3]], b=[domain[2] domain[4]])
k = WassersteinDictionaries.get_gibbs_matrix(c, ε)
MC = MatrixCache(N_fine + 1)
ρ(u) = u
ρ̂_train = [get_ρ̂(u, ρ, V_fine, N_fine + 1) for u in u_train]
SP = SinkhornParameters(Int(10 * ceil(1 / ε)), ε, 1e-3, false, debias, true)
ρ̂_ref = sinkhorn_barycenter_sep([1 / nₛ for _ in ρ̂_train], ρ̂_train, k, SP, MC)
log_ρ̂_ref = safe_log.(ρ̂_ref);
ρ_ref = interpolate_everywhere(Interpolable(FEFunction(V_fine, vec(ρ̂_ref))), Ψ);

### Transport
println("Calculating transport potentials...")
ψ̂ᶜ = [get_ψ̂_ψ̂ᶜ(ρ̂, ρ̂_ref, k, SP, MC)[2] for ρ̂ in ρ̂_train];
ψᶜ = [boundary_projection(ψᶜ, δ⁻¹, κ, Ψ, dΩ, 2 * highorder) for ψᶜ in ψ̂ᶜ]

println("Calculating RB of transport potentials...")
ξᶜ, evd_ψᶜ = pod_monge_embedding(ψᶜ, ρ_ref, Ψ, Ψ, dΩ, τ)
m = length(ξᶜ)
println("Found $m transport modes.")


λ = [[sum(∫(∇(_ψᶜ) ⋅ ∇(_ξᶜ) * ρ_ref)dΩ) for _ξᶜ in ξᶜ] for _ψᶜ in ψᶜ]
ψᶜ_train = [FEFunction(Ψ, _λ' * get_free_dof_values.(ξᶜ)) for _λ in λ]
μ_mat = zeros(2, length(ω_train_all))
for i in eachindex(ω_train_all)
    μ_mat[2, i] = ω_train_all[i]
    μ_mat[1, i] = t_train_all[i]
end
μ_mat
gp = get_gp(μ_mat, λ, m);

println("Mapping snapshots...")
T★u_train = [pushfwd(u_train[i], ψᶜ_train[i], V₂, dΩ) for i in eachindex(u_train)];

println("Calculating POD-RB after remapping...")
ϕ, evd_T★u = pod(T★u_train, V₂, U₀, dΩ, τ)
nₘ = length(ϕ)
println("Found POD-RB of size $nₘ after registration.")

println("Performing the offline EIM calculations for the mapped RB...")
J = vec([get_J(get_transport_potential([t, ω], ξᶜ, Ψ, gp), W) for ω in ω_train, t in t_train]);
Ξ_J, evd_J = pod(J, W, W, dΩ, τ_eim)
eim_J = EmpiricalInterpolation(Ξ_J, (ϕ, X) -> form_J(ϕ, X, dΩ), ϕ, W)
Q_J = get_Q(eim_J)
K★J = vec([get_K(get_transport_potential([t, ω], ξᶜ, Ψ, gp), W_matrix) for t in t_train, ω in ω_train]);
Ξ_K, evd_K = pod(K★J, W_matrix, W_matrix, dΩ, τ_eim);
eim_K = EmpiricalInterpolation(Ξ_K, (ϕ, X) -> form_K(ϕ, X, dΩ), ϕ, W_matrix);
Q_K = get_Q(eim_K)
∂tΦ★J = vec([get_∂tΦ★J(get_transport_potential_∂t(t, ω, dt, ξᶜ, Ψ, gp),
    get_transport_potential([t, ω], ξᶜ, Ψ, gp),
    W_vector) for ω in ω_train, t in t_train]);
Ξ_∂tΦ★J, evd_∂tΦ★J = pod(∂tΦ★J, W_vector, W_vector, dΩ, τ_eim);
eim_∂tΦ★J = EmpiricalInterpolation(Ξ_∂tΦ★J, (ϕ, X) -> form_∂tΦ★J(ϕ, X, dΩ), ϕ, W_vector)
Q_∂tΦ★J = get_Q(eim_∂tΦ★J)
ā★J = vec([get_ā★J(w̄, ω, get_transport_potential([t, ω], ξᶜ, Ψ, gp), W_vector) for ω in ω_train, t in t_train]);
Ξ_ā★J, evd_ā★J = pod(ā★J, W_vector, W_vector, dΩ, τ_eim);
eim_ā★J = EmpiricalInterpolation(Ξ_ā★J, (ϕ, X) -> form_ā★J(ϕ, X, dΩ), ϕ, W_vector);
Q_ā★J = get_Q(eim_ā★J)
eim_āu★J = EmpiricalInterpolation(Ξ_ā★J, (ϕ, X) -> form_āu★J(ϕ, X, dΩ), ϕ, W_vector);
Q_āu★J = get_Q(eim_āu★J)
Q_āu★J, Q_ā★J, Q_∂tΦ★J, Q_K, Q_J

ũ_trb_eim = snapshots(nu₀, ω_test, ϕ, V₂, U₂, dΩ, ξᶜ, gp, Ψ, nls, θ, t0, dt, tF_test,
    eim_J, eim_∂tΦ★J, eim_ā★J, eim_K, eim_āu★J, P)

T★u_trb_eim = [FEFunction(V₂, u' * get_free_dof_values.(ϕ)) for u in ũ_trb_eim];

ψᶜ_test = vec([get_transport_potential([t, ω], ξᶜ, Ψ, gp) for t in t0:dt:tF_test, ω in ω_test]);
ψ̂_test = [c_transform(ψᶜ, V_fine, c, log_ρ̂_ref, MC, 0.1 * 1 / N^2) for ψᶜ in ψᶜ_test];
ψ_test = [boundary_projection(ψ, δ⁻¹, κ, Ψ, dΩ, 2 * highorder) for ψ in ψ̂_test] # performing a projection on the boundary

u_trb_eim = [pushfwd(T★u_trb_eim[i], ψ_test[i], V₂, dΩ) for i in eachindex(T★u_trb_eim)];

ΔL2_rb_eim = rel_error_vec(u_rb_eim, u_test, L2, dΩ)
ΔH1_rb_eim = rel_error_vec(u_rb_eim, u_test, H1, dΩ)

ΔL2_trb_eim = rel_error_vec(u_trb_eim, u_test, L2, dΩ)
ΔH1_trb_eim = rel_error_vec(u_trb_eim, u_test, H1, dΩ);

M_ΔL2_rb = reshape(ΔL2_rb_eim, :, nₒₜ)'
M_ΔL2_trb = reshape(ΔL2_trb_eim, :, nₒₜ)'
M_ΔH1_rb = reshape(ΔH1_rb_eim, :, nₒₜ)'
M_ΔH1_trb = reshape(ΔH1_trb_eim, :, nₒₜ)';

### Results

println(" ")
println("ε = $ε, τ = $τ")
# L2 errors
@printf "unregistered (n = %.0f, m = %.0f) \t L2 error avg.: %.2e ± %.2e \t max.: %.2e \n" length(ζ) 0 Statistics.mean(M_ΔL2_rb[:, end]) Statistics.std(M_ΔL2_rb[:, end]) maximum(M_ΔL2_rb[:, end])
@printf "registered (n = %.0f, m = %.0f) \t L2 error avg.: %.2e ± %.2e \t max.: %.2e \n" nₘ m Statistics.mean(M_ΔL2_trb[:, end]) Statistics.std(M_ΔL2_trb[:, end]) maximum(M_ΔL2_trb[:, end])
println(" ")

### Plots

#=
# plot the L2 error over time
cpal = palette(:thermal, 4)
plot(t0:dt:tF_test, Statistics.mean(M_ΔL2_trb, dims=1)', minorgrid=true, ylim=(0, 0.5),
    ribbon=(Statistics.mean(M_ΔL2_trb, dims=1)' - minimum(M_ΔL2_trb, dims=1)',
        -Statistics.mean(M_ΔL2_trb, dims=1)' + maximum(M_ΔL2_trb, dims=1)'),
    linewidth=2, color=cpal[2], label=L"u_{\mathrm{trb,eim}}", dpi=400, xticks=0:0.2:1)
plot!(t0:dt:tF_test, Statistics.mean(M_ΔL2_rb, dims=1)',
    ribbon=(Statistics.mean(M_ΔL2_rb, dims=1)' - minimum(M_ΔL2_rb, dims=1)',
        -Statistics.mean(M_ΔL2_rb, dims=1)' + maximum(M_ΔL2_rb, dims=1)'),
    linewidth=2, color=cpal[3], xlabel=L"t", ylabel=L"\Vert u - u_{\{\mathrm{rb,trb}\}} \Vert_{L^2} / \Vert u \Vert_{L^2}",
    legend=:topleft, label=L"u_{\mathrm{rb, eim}}",
    legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12
)
vline!([0.8], color="grey", style=:dash, label=false)
#savefig("figs/errors.png")

### plot some snapshot crossections before and after registration
cpal = palette(:thermal, 17);
plt1 = plot()
_i = 1
idxs = rand(1:length(u_train),15)
for i in idxs
    plot!(0:0.001:1, x -> T★u_train[i](Point(x, 0.5)), legend=false,
        linewidth=2, color=cpal[_i], xlabel=L"u", ylabel=L"u \circ \Phi^{-1}",
        legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12)
    global _i += 1
end
plt2 = plot()
_i = 1
for i in idxs
    plot!(0:0.001:1, x -> u_train[i](Point(x, 0.5)), legend=false,
        linewidth=2, color=cpal[_i], xlabel=L"x", ylabel=L"u",
        legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12)
    global _i += 1
end
plot(plt1, plt2)

# plot ev decay"
cpal = palette(:thermal, 4);
plot(abs.(evd_u.values) ./ evd_u.values[1], yaxis=:log, minorgrid=true, xaxis=:log,
    yticks=10.0 .^ (-16:2:0), xticks=([1, 10, 100], string.([1, 10, 100])),
    linewidth=2, marker=:circle, xlabel=L"n", ylabel=L"\lambda_n / \lambda_1", ylim=(1e-16, 2), label=L"u", color=cpal[1],
    legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, legend=:bottomleft)
plot!(abs.(evd_T★u.values) ./ evd_T★u.values[1], linewidth=2, marker=:square, markersize=3, label=L"u \circ \Phi^{-1}", color=cpal[2])
plot!(abs.(evd_ψᶜ.values) ./ evd_ψᶜ.values[1], linewidth=2, marker=:diamond, label=L"\nabla \psi^c", color=cpal[3])
savefig("figs/evds_log.png")

nₚ = minimum((nₛ, 50))
cpal = palette(:thermal, 6)
plot(abs.(evd_J.values[1:nₚ]) ./ evd_J.values[1], yaxis=:log, minorgrid=true, xaxis=:log, legend=:bottomleft,
    yticks=10.0 .^ (-16:2:0), xticks=([1, 10, 50], string.([1, 10, 50])),
    linewidth=2, marker=:circle, xlabel=L"n", ylabel=L"\lambda_n / \lambda_1", ylim=(1e-16, 2), label=L"\det D\Phi^{-1}", color=cpal[1],
    legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, dpi=400)
plot!(abs.(evd_∂tΦ★J.values[1:nₚ]) ./ evd_∂tΦ★J.values[1], linewidth=2, marker=:square, markersize=3,
    label=L"D\Phi^{-1} \partial_t \Phi^{-1} \det D\Phi^{-1}", color=cpal[2])
plot!(abs.(evd_ā★J.values[1:nₚ]) ./ evd_ā★J.values[1], linewidth=2, marker=:diamond,
    label=L"D\Phi^{-1} \bar a \, \det D\Phi^{-1}", color=cpal[3])
plot!(abs.(evd_K.values[1:nₚ]) ./ evd_K.values[1], linewidth=2, marker=:utriangle,
    label=L"[D\Phi^{-1}]^{-1} [D\Phi^{-1}]^{-T} \det D\Phi^{-1}", color=cpal[4])
savefig("figs/evds_log_eim.png")

_i = argmax(M_ΔL2_trb[:, end])
plt1 = plot()
s_fine = -0.5:0.005:0.5
cpal = palette(:thermal, 4)
for i in 1 + (_i - 1) * length(t0:0.05:tF_test)
    plot!(s_fine, x -> (u_test[i])(Point([0.5, 0.5] + x .* [cos(ω_test[_i]), sin(ω_test[_i])])),
        xlabel=L"s", label=L"u",
        linewidth=2, color=cpal[1],
        legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, dpi=400, legend=:topleft)
    plot!(s_fine, x -> (u_trb_eim[i])(Point([0.5, 0.5] + x .* [cos(ω_test[_i]), sin(ω_test[_i])])),
        linewidth=2, color=cpal[2], label=L"u_{\mathrm{trb,eim}}",
        legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12)
    plot!(s_fine, x -> (u_rb_eim[i])(Point([0.5, 0.5] + x .* [cos(ω_test[_i]), sin(ω_test[_i])])),
        linewidth=2, color=cpal[3], label=L"u_{\mathrm{rb,eim}}",
        legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12)
    println(i)
end
for i in ((_i - 1) * length(t0:0.05:tF_test) + div(1 + length(t0:0.05:tF_test), 2), (_i - 1) * length(t0:0.05:tF_test) + length(t0:0.05:tF_test))
    plot!(s_fine, x -> (u_test[i])(Point([0.5, 0.5] + x .* [cos(ω_test[_i]), sin(ω_test[_i])])),
        xlabel=L"s", label=false,
        linewidth=2, color=cpal[1],
        legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12)
    plot!(s_fine, x -> (u_trb_eim[i])(Point([0.5, 0.5] + x .* [cos(ω_test[_i]), sin(ω_test[_i])])),
        linewidth=2, color=cpal[2], label=false,
        legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12)
    plot!(s_fine, x -> (u_rb_eim[i])(Point([0.5, 0.5] + x .* [cos(ω_test[_i]), sin(ω_test[_i])])),
        linewidth=2, color=cpal[3], label=false,
        legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12)
    println(i)
end
plt1
savefig("figs/worstcase.png")
=$