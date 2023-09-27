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

### Setup

# hyperparameters
const ε = 1e-2
const τ = 1e-4
const τ_eim = 0.1τ

const δ⁻¹ = 1e9
const κ = 1 / sqrt(ε)
const debias = true

# parameters of the PPDE problem
const nₛ = 20
const nₜ = 10

# parameter space
const μ_min = -0.35
const μ_max = 0.35

# finite element spaces
const d = 2
const N = 32
const highorder = 3
const N_fine = highorder * N
const ε_fine = 0.1 / N^2

P = PoissonProblem(1e-3);
f(x, x0) = exp(-((x[1] - x0[1])^2 + (x[2] - x0[2])^2) / 2 / P.var) / (2π * P.var)
fe_spaces = initialize_fe_spaces(N, N_fine, d, highorder, P)
dΩ = fe_spaces.dΩ

### Training set

Random.seed!(1234)
μ_train = testparameterset(nₛ, μ_max, μ_min)
println("Calculating training snapshots...")
uE_train = [snapshot(f, μ, fe_spaces.V, fe_spaces.U, dΩ, P) for μ in μ_train]
u_train = [uE[1] for uE in uE_train]
E_train = [uE[2] for uE in uE_train];

### Reduced basis

println("Calculating POD-RB...")
ζ, evd_u = pod(u_train, fe_spaces.V, fe_spaces.U, dΩ, τ)
n = length(ζ)
println("Found POD-RB of size $n.")
Aᵣ = OptimalMappings.get_A(μ_train[1], ζ, dΩ);

### Test set

μ_test = testparameterset(nₜ, μ_max, μ_min)
println("Calculating test snapshots...")
uE_test = [snapshot(f, μ, fe_spaces.V, fe_spaces.U, dΩ, P) for μ in μ_test]
u_test = [uE[1] for uE in uE_test]
E_test = [uE[2] for uE in uE_test]
println("Solving RB problem...") # μ - independent
ũE_rb = [snapshot(f, μ, fe_spaces.V, fe_spaces.U, dΩ, ζ, Aᵣ, P) for μ in μ_test]
ũ_rb = [uE[1] for uE in ũE_rb]
E_rb = [uE[2] for uE in ũE_rb]
u_rb = [FEFunction(fe_spaces.V, u' * get_free_dof_values.(ζ)) for u in ũ_rb]

### Transport calculations'

println("Calculating densities...")
c = WassersteinDictionaries.get_cost_matrix_separated(N_fine + 1, d, a=[fe_spaces.domain[1] fe_spaces.domain[3]], b=[fe_spaces.domain[2] fe_spaces.domain[4]])
k = WassersteinDictionaries.get_gibbs_matrix(c, ε)
MC = MatrixCache(N_fine + 1)
ρ(u) = u ⋅ u
#ρ̂_train = [get_ρ̂(x -> f(x, μ .+ 0.5), fe_spaces.V_fine, N_fine + 1) for μ in μ_train]
ρ̂_train = [get_ρ̂(u, ρ, fe_spaces.V_fine, N_fine + 1) for u in u_train]
SP = SinkhornParameters(Int(10 * ceil(1 / ε)), ε, 1e-3, false, debias, true)
ρ̂_ref = sinkhorn_barycenter_sep([1 / nₛ for _ in ρ̂_train], ρ̂_train, k, SP, MC)
log_ρ̂_ref = safe_log.(ρ̂_ref)
ρ_ref = interpolate_everywhere(Interpolable(FEFunction(fe_spaces.V_fine, vec(ρ̂_ref))), fe_spaces.Ψ);

### Transport maps

println("Calculating transport potentials...")
ψ̂ᶜ = [get_ψ̂_ψ̂ᶜ(ρ̂, ρ̂_ref, k, SP, MC)[2] for ρ̂ in ρ̂_train];
ψᶜ = [boundary_projection(ψᶜ, δ⁻¹, κ, fe_spaces.Ψ, dΩ, 2 * highorder) for ψᶜ in ψ̂ᶜ];

### Transport modes

println("Calculating RB of transport potentials...")
ξᶜ, evd_ψᶜ = pod_monge_embedding(ψᶜ, ρ_ref, fe_spaces.Ψ, fe_spaces.Ψ, dΩ, τ)
m = length(ξᶜ)
println("Found $m transport modes.")

### Gaussian process

w = [[sum(∫(∇(_ψᶜ) ⋅ ∇(_ξᶜ) * ρ_ref)dΩ) for _ξᶜ in ξᶜ] for _ψᶜ in ψᶜ]
ψᶜ_train = [FEFunction(fe_spaces.Ψ, _w' * get_free_dof_values.(ξᶜ)) for _w in w]
μ_mat = [μ[k] for k in 1:d, μ in μ_train]
gp = get_gp(μ_mat, w, m);

### Reference reduced basis

println("Mapping snapshots...")
T★u_train = [pushfwd(u_train[i], ψᶜ_train[i], fe_spaces.V, dΩ) for i in eachindex(u_train)];
println("Calculating POD-RB after remapping...")
ϕ, evd_T★u = pod(T★u_train, fe_spaces.V, fe_spaces.U, dΩ, τ)
nₘ = length(ϕ)
println("Found POD-RB of size $nₘ after registration.")

### Online (no EIM)

println("Solving registered RB problem...")
ψᶜ_test = [get_transport_potential(μ, ξᶜ, fe_spaces.Ψ, gp) for μ in μ_test];
ũẼ_trb = [snapshot(f, μ_test[i], fe_spaces.V, fe_spaces.U, dΩ, ϕ, ψᶜ_test[i], P) for i in eachindex(μ_test)];
T★u_trb = [FEFunction(fe_spaces.V, uE[1]' * get_free_dof_values.(ϕ)) for uE in ũẼ_trb];
E_trb = [uE[2] for uE in ũẼ_trb];

### Remapping

println("Inverting registration maps and enforcing boundary conditions...")
ψ̂_test = [c_transform(ψᶜ, fe_spaces.V_fine, c, log_ρ̂_ref, MC, ε_fine) for ψᶜ in ψᶜ_test];
ψ_test = [boundary_projection(ψ, δ⁻¹, κ, fe_spaces.Ψ, dΩ, 2 * highorder) for ψ in ψ̂_test]
println("Remapping the registered RB approximations...")
u_trb = [pushfwd(T★u_trb[i], ψ_test[i], fe_spaces.V, dΩ) for i in eachindex(μ_test)];
T★u_test = [pushfwd(u_test[i], ψᶜ_test[i], fe_spaces.V, dΩ) for i in eachindex(μ_test)];

### Empirical interpolations

f_nomap = [interpolate_everywhere(x -> f(x, μ .+ 0.5), fe_spaces.W) for μ in μ_train]
Ξ_f, evd_f = pod(f_nomap, fe_spaces.W, fe_spaces.W, dΩ, τ_eim);
Q_f = length(Ξ_f)
println("Performing the offline EIM calculations...")
f★J = [get_f★J(f, μ, get_transport_potential(μ, ξᶜ, fe_spaces.Ψ, gp), fe_spaces.W) for μ in μ_train]
Ξ_f★J, evd_f★J = pod(f★J, fe_spaces.W, fe_spaces.W, dΩ, τ_eim)
eim_f★J = EmpiricalInterpolation(Ξ_f★J, (ϕ,X) -> form_f★J(ϕ,X,dΩ), ϕ, fe_spaces.W)
Q_f★J = length(Ξ_f★J)
K = [get_K(get_transport_potential(μ, ξᶜ, fe_spaces.Ψ, gp), fe_spaces.W_matrix) for μ in μ_train];
Ξ_K, evd_K = pod(K, fe_spaces.W_matrix, fe_spaces.W_matrix, dΩ, τ_eim)
eim_K = EmpiricalInterpolation(Ξ_K, (ϕ,X) -> form_K(ϕ,X,dΩ), ϕ, fe_spaces.W_matrix);
Q_K = length(Ξ_K)
println("Found EIM approximations with Q = $Q_f★J and $Q_K.")

### Online phase

println("Solving registered RB problem using EIM...")
ũE_trb_eim = [snapshot(f, μ_test[i], fe_spaces.V, fe_spaces.U, dΩ, ϕ, ψᶜ_test[i], eim_f★J, eim_K, P) for i in eachindex(μ_test)];
T★u_trb_eim = [FEFunction(fe_spaces.V, uE[1]' * get_free_dof_values.(ϕ)) for uE in ũE_trb_eim];
E_trb_eim = [uE[2] for uE in ũE_trb_eim];

println("Remapping the registered RB approximations...")
u_trb_eim = [pushfwd(T★u_trb_eim[i], ψ_test[i], fe_spaces.V, dΩ) for i in eachindex(μ_test)]

### Errors

ΔL2_rb = rel_error_vec(u_rb, u_test, L2, dΩ)
ΔH1_rb = rel_error_vec(u_rb, u_test, H1, dΩ)
ΔE_rb = rel_error_vec(E_rb, E_test, (u, dΩ) -> abs(u), dΩ)

ΔL2_trb = rel_error_vec(u_trb, u_test, L2, dΩ)
ΔH1_trb = rel_error_vec(u_trb, u_test, H1, dΩ)
ΔE_trb = rel_error_vec(E_trb, E_test, (u, dΩ) -> abs(u), dΩ)

ΔL2_trb_ref = rel_error_vec(T★u_trb, T★u_test, L2, dΩ)
ΔH1_trb_ref = rel_error_vec(T★u_trb, T★u_test, H1, dΩ)

ΔL2_trb_eim = rel_error_vec(u_trb_eim, u_test, L2, dΩ)
ΔH1_trb_eim = rel_error_vec(u_trb_eim, u_test, H1, dΩ)
ΔE_trb_eim = rel_error_vec(E_trb_eim, E_test, (u, dΩ) -> abs(u), dΩ)

ΔL2_trb_ref_eim = rel_error_vec(T★u_trb_eim, T★u_test, L2, dΩ)
ΔH1_trb_ref_eim = rel_error_vec(T★u_trb_eim, T★u_test, H1, dΩ)

Ḣ1_ρ_ψ = rel_error_vec(ψᶜ_train, ψᶜ, (u, dΩ) -> Ḣ1(u, ρ_ref, dΩ), dΩ);

### Results

println(" ")
println("ε = $ε, τ = $τ")
# L2 errors
@printf "unregistered (n = %.0f, m = %.0f) \t L2 error avg.: %.2e ± %.2e \t max.: %.2e \n" n 0 Statistics.mean(ΔL2_rb) Statistics.std(ΔL2_rb) maximum(ΔL2_rb)
@printf "registered (n = %.0f, m = %.0f) \t L2 error avg.: %.2e ± %.2e \t max.: %.2e \n" nₘ m Statistics.mean(ΔL2_trb) Statistics.std(ΔL2_trb) maximum(ΔL2_trb)
@printf "with EIM (Qf = %.0f, QK = %.0f) \t L2 error avg.: %.2e ± %.2e \t max.: %.2e \n" Q_f★J Q_K Statistics.mean(ΔL2_trb_eim) Statistics.std(ΔL2_trb_eim) maximum(ΔL2_trb_eim)
@printf "in reference domain: \t \t L2 error avg.: %.2e ± %.2e \t max.: %.2e \n" Statistics.mean(ΔL2_trb_ref_eim) Statistics.std(ΔL2_trb_ref_eim) maximum(ΔL2_trb_ref_eim)
println(" ")
@printf "unregistered (n = %.0f, m = %.0f) \t H1 error avg.: %.2e ± %.2e \t max.: %.2e \n" n 0 Statistics.mean(ΔH1_rb) Statistics.std(ΔH1_rb) maximum(ΔH1_rb)
@printf "registered (n = %.0f, m = %.0f) \t H1 error avg.: %.2e ± %.2e \t max.: %.2e \n" nₘ m Statistics.mean(ΔH1_trb) Statistics.std(ΔH1_trb) maximum(ΔH1_trb)
@printf "with EIM (Qf = %.0f, QK = %.0f) \t H1 error avg.: %.2e ± %.2e \t max.: %.2e \n" Q_f★J Q_K Statistics.mean(ΔH1_trb_eim) Statistics.std(ΔH1_trb_eim) maximum(ΔH1_trb_eim)
@printf "in reference domain: \t \t H1 error avg.: %.2e ± %.2e \t max.: %.2e \n" Statistics.mean(ΔH1_trb_ref_eim) Statistics.std(ΔH1_trb_ref_eim) maximum(ΔH1_trb_ref_eim)
println(" ")
@printf "unregistered (n = %.0f, m = %.0f) \t En. err. avg.: %.2e ± %.2e \t max.: %.2e \n" n 0 Statistics.mean(ΔE_rb) Statistics.std(ΔE_rb) maximum(ΔE_rb)
@printf "registered (n = %.0f, m = %.0f) \t En. err. avg.: %.2e ± %.2e \t max.: %.2e \n" nₘ m Statistics.mean(ΔE_trb) Statistics.std(ΔE_trb) maximum(ΔE_trb)
@printf "with EIM (Qf = %.0f, QK = %.0f) \t En. err. avg.: %.2e ± %.2e \t max.: %.2e \n" Q_f★J Q_K Statistics.mean(ΔE_trb_eim) Statistics.std(ΔE_trb_eim) maximum(ΔE_trb_eim)
println(" ")

### Plots
#=

# plot the worst case (H1 error) reconstruction
_i = argmax(ΔH1_trb_eim)
μ = μ_test[_i]
cpal = palette(:thermal, 5)
plt1 = plot(0:0.005:1, x -> u_test[_i](Point(x, 0.5 + μ_test[_i][2])), linewidth=2, color=cpal[1], xlabel=L"x",
    label=L"u", legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, dpi=400)
plot!(0:0.005:1, x -> u_rb[_i](Point(x, 0.5 + μ_test[_i][2])), linewidth=2, color=cpal[2], label=L"u_{\mathrm{rb}}")
plot!(0:0.005:1, x -> u_trb[_i](Point(x, 0.5 + μ_test[_i][2])), linewidth=2, color=cpal[3], label=L"u_{\mathrm{trb}}")
plot!(0:0.005:1, x -> u_trb_eim[_i](Point(x, 0.5 + μ_test[_i][2])), linewidth=2, color=cpal[4], label=L"u_{\mathrm{trb, eim}}")
plt2 = plot(0:0.005:1, x -> u_test[_i](Point(0.5 + μ_test[_i][1], x)), linewidth=2, color=cpal[1], xlabel=L"y",
    label=L"u", legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, dpi=400)
plot!(0:0.005:1, x -> u_rb[_i](Point(0.5 + μ_test[_i][1], x)), linewidth=2, color=cpal[2], label=L"u_{\mathrm{rb}}")
plot!(0:0.005:1, x -> u_trb[_i](Point(0.5 + μ_test[_i][1], x)), linewidth=2, color=cpal[3], label=L"u_{\mathrm{trb}}")
plot!(0:0.005:1, x -> u_trb_eim[_i](Point(0.5 + μ_test[_i][1], x)), linewidth=2, color=cpal[4], label=L"u_{\mathrm{trb, eim}}")

plot(plt1, plt2)
savefig("figs/crosssecs.png")

# plot ev decay
cpal = palette(:thermal, 4);
plot(abs.(evd_f.values) ./ evd_f.values[1], yaxis=:log, minorgrid=true, xaxis=:log,
    yticks=10.0 .^ (-16:2:0), xticks=([1, 10, 100], string.([1, 10, 100])),
    linewidth=2, marker=:circle, xlabel=L"n", ylabel=L"\lambda_n / \lambda_1", ylim=(1e-16, 2), label=L"f", color=cpal[1],
    legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, legend=:bottomleft)
plot!(abs.(evd_f★J.values) ./ evd_f★J.values[1], linewidth=2, marker=:square, markersize=3, label=L"f \circ \Phi^{-1} \det D\Phi^{-1}", color=cpal[2])
plot!(abs.(evd_K.values) ./ evd_K.values[1], linewidth=2, marker=:diamond, label=L"[D\Phi^{-1}]^{-1} [D\Phi^{-1}]^{-T} \det D\Phi^{-1}", color=cpal[3])
savefig("figs/evds_eim_log.png")

println("Plotting transport modes...")
x_cord = 0:0.025:1
for i in 1:m
        surface(x_cord, x_cord, (x, y) -> ξᶜ[i](Point(x, y)) - ξᶜ[i](Point(0.5, 0.5)),
        cbar=false, xlabel=L"x", ylabel=L"y", zlabel=L"\xi^c_{%$i}", size=(400, 400), camera=(30, 30),
        legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, zguidefontsize=12, dpi=400)
    savefig("figs/xi$i.png")
end

# plot gp
for i in 1:m
    surface(gp[i]; legend=false, xlabel=L"\mu_1", ylabel=L"\mu_2", zlabel=L"w_{%$i}",
        xlim=(μ_min, μ_max), ylim=(μ_min, μ_max), camera=(30, 45), size=(400, 400),
        legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, zguidefontsize=12, dpi=400)
    scatter!([μ[1] for μ in μ_train], [μ[2] for μ in μ_train], predict_y(gp[i], μ_mat)[1], label=false, color=:white, markersize=2, aspectratio=1)
    savefig("figs/w$i.png")
end

# plot ev decay"
cpal = palette(:thermal, 4);
plot(abs.(evd_u.values) ./ evd_u.values[1], yaxis=:log, minorgrid=true, xaxis=:log,
    yticks=10.0 .^ (-16:2:0), xticks=([1, 10, 100], string.([1, 10, 100])),
    linewidth=2, marker=:circle, xlabel=L"n", ylabel=L"\lambda_n / \lambda_1", ylim=(1e-16, 2), label=L"u", color=cpal[1],
    legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, legend=:bottomleft)
plot!(abs.(evd_T★u.values) ./ evd_T★u.values[1], linewidth=2, marker=:square, markersize=3, label=L"u \circ \Phi^{-1}", color=cpal[2])
plot!(abs.(evd_ψᶜ.values) ./ evd_ψᶜ.values[1], linewidth=2, marker=:diamond, label=L"\nabla \psi^c", color=cpal[3])
savefig("figs/evds_log.png")

### plot some snapshot crossections before and after registration
cpal = palette(:thermal, 11);
plt1 = plot()
_i = 1
idxs = 1:10
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
savefig("figs/u_registered.png")

=#