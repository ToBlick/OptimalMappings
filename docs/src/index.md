# OptimalMappings.jl

This package contains the code used in the numerical examples of [arXiv:2304.14884](https://arxiv.org/abs/2304.14884). It relies on [Gridap.jl](https://github.com/gridap/Gridap.jl) for finite element routines and [OptimalTransportTools.jl](https://github.com/JuliaRCM/OptimalTransportTools.jl) for computational optimal transport.

## Example 1

The equation to solve reads 
``\Delta u(x; \mu) = f(x; \mu) : x \in \Omega, u(x; \mu) = 0 : x \in \partial \Omega.``

```@setup 1
using OptimalMappings
using Printf
using LaTeXStrings
using Random
using WassersteinDictionaries
using Gridap, Gridap.FESpaces, Gridap.CellData
using Gridap.CellData: get_cell_quadrature, get_node_coordinates
using LineSearches
using Plots
using GaussianProcesses
using Statistics
```
### Setup
We begin by setting the hyperparameters: the entropic regulaization strength and the eigenvalue energy tolerance being the most important.

```@example 1
const ε = 1e-2
const τ = 1e-4
const τ_eim = 0.1τ
const δ⁻¹ = 1e9
const κ = 1 / sqrt(ε)
const debias = true
nothing # hide
```

Then, we define the PPDE problem and set up the needed FE spaces.

```@example 1
const nₛ = 50
const nₜ = 25
const μ_min = -0.35
const μ_max = 0.35
const var = 1e-3
P = PoissonProblem(var)
f(x, x0) = exp(-((x[1] - x0[1])^2 + (x[2] - x0[2])^2) / 2 / P.var) / (2π * P.var)
nothing # hide
```

```@example 1
const d = 2
const N = 32
const highorder = 3
const N_fine = highorder * N
const ε_fine = 0.1 / N^2
fe_spaces = initialize_fe_spaces(N, N_fine, d, highorder, P)
dΩ = fe_spaces.dΩ
nothing # hide
```

### Training and testing data

We compute the training snapshots ``u(\mu_i) \; \forall i = 1, \dots, n_s``

```@example 1
Random.seed!(1234) # hide
μ_train = testparameterset(nₛ, μ_max, μ_min)
uE_train = [snapshot(f, μ, fe_spaces.V, fe_spaces.U, dΩ, P) for μ in μ_train]
u_train = [uE[1] for uE in uE_train]
E_train = [uE[2] for uE in uE_train]
nothing # hide
```

and the test set.

```@example 1
μ_test = testparameterset(nₜ, μ_max, μ_min)
uE_test = [snapshot(f, μ, fe_spaces.V, fe_spaces.U, dΩ, P) for μ in μ_test]
u_test = [uE[1] for uE in uE_test]
E_test = [uE[2] for uE in uE_test]
nothing #hide
```

### Reduced basis

The reduced basis without registration ``\zeta_1, \dots, \zeta_n``

```@example 1
ζ, evd_u = pod(u_train, fe_spaces.V, fe_spaces.U, dΩ, τ)
n = length(ζ)
```
is used solve the reduced problem without registration.

```@example 1
Aᵣ = OptimalMappings.get_A(μ_train[1], ζ, dΩ)
ũE_rb = [snapshot(f, μ, fe_spaces.V, fe_spaces.U, dΩ, ζ, Aᵣ, P) for μ in μ_test]
ũ_rb = [uE[1] for uE in ũE_rb]
E_rb = [uE[2] for uE in ũE_rb]
u_rb = [FEFunction(fe_spaces.V, u' * get_free_dof_values.(ζ)) for u in ũ_rb]
nothing # hide
```

### Optimal transport calculations

The choice for ``\rho`` is ``\rho(\mu) := \tfrac{u(\mu)^2}{\int u(\mu)^2}``. The reference density ``\bar \rho`` is the (unweighted) OT barycenter. 

```@example 1
c = WassersteinDictionaries.get_cost_matrix_separated(N_fine+1, d, a=[fe_spaces.domain[1] fe_spaces.domain[3]], b=[fe_spaces.domain[2] fe_spaces.domain[4]])
k = WassersteinDictionaries.get_gibbs_matrix(c, ε)
MC = MatrixCache(N_fine + 1)
ρ(u) = u ⋅ u
ρ̂_train = [get_ρ̂(u, ρ, fe_spaces.V_fine, N_fine + 1) for u in u_train]
SP = SinkhornParameters(Int(10 * ceil(1 / ε)), ε, 1e-3, false, debias, true)
ρ̂_ref = sinkhorn_barycenter_sep([1 / nₛ for _ in ρ̂_train], ρ̂_train, k, SP, MC)
log_ρ̂_ref = safe_log.(ρ̂_ref)
ρ_ref = interpolate_everywhere(Interpolable(FEFunction(fe_spaces.V_fine, vec(ρ̂_ref))), fe_spaces.Ψ);
nothing # hide
```

### Transport modes

Next, the transport potentials ``\psi^c_i`` between ``\bar \rho`` and all ``\rho(\mu_i)`` are computed. The boundary projection guarantees that ``y \mapsto y - \nabla \psi^c_i(y)`` is orthogonal to the domain boundary.

```@example 1
ψ̂ᶜ = [get_ψ̂_ψ̂ᶜ(ρ̂, ρ̂_ref, k, SP, MC)[2] for ρ̂ in ρ̂_train];
ψᶜ = [boundary_projection(ψᶜ, δ⁻¹, κ, fe_spaces.Ψ, dΩ, 2 * highorder) for ψᶜ in ψ̂ᶜ]
nothing # hide
```

The transport modes ``\xi^c_j`` are obtained by performing a proper orthogonal decomposition on the transport maps as elements of the tangent space of ``\mathcal P`` at ``\bar \rho``.

```@example 1
ξᶜ, evd_ψᶜ = pod_monge_embedding(ψᶜ, ρ_ref, fe_spaces.Ψ, fe_spaces.Ψ, dΩ, τ)
m = length(ξᶜ)
```

```@setup 1
x_cord = 0:0.025:1
ξᶜs = []
for i in 1:minimum((4,m))
    push!(ξᶜs, 
        surface(x_cord, x_cord, (x, y) -> ξᶜ[i](Point(x, y)) - ξᶜ[i](Point(0.5, 0.5)),
        cbar=false, xlabel=L"x", ylabel=L"y", zlabel=L"\xi^c_{%$i}")
        )
end
```
``` @example 1
plot(ξᶜs...)
```

### The ``\mu \mapsto \Phi^{-1}_\mu`` map

The mapping ``\Phi^{-1}_\mu: y \mapsto y - \sum_j w(\mu)_j \nabla \xi^c_j(y)`` is determined by the weights ``w_j(\mu)``, which are the output of a Gaussian process, fitted to the training data.

```@example 1
w = [[sum(∫(∇(_ψᶜ) ⋅ ∇(_ξᶜ) * ρ_ref)dΩ) for _ξᶜ in ξᶜ] for _ψᶜ in ψᶜ]
ψᶜ_train = [FEFunction(fe_spaces.Ψ, _w' * get_free_dof_values.(ξᶜ)) for _w in w]
μ_mat = [μ[k] for k in 1:d, μ in μ_train]
gp = get_gp(μ_mat, w, m);
nothing # hide
```

```@setup 1
plt = plot()
ws = []
for i in 1:minimum((4,m))
    _s = surface(gp[i]; legend=false, xlabel=L"\mu_1", ylabel=L"\mu_2", zlabel=L"w_{%$i}", xlim=(μ_min, μ_max), ylim=(μ_min, μ_max))
    scatter!([μ[1] for μ in μ_train], [μ[2] for μ in μ_train], predict_y(gp[i], μ_mat)[1], label=false, color=:white, markersize=2, aspectratio=1)
    push!(ws, _s)
end
```
```@example 1
plot(ws...)
```

### Reference reduced basis

With the constructed mappings ``\Phi^{-1}_\mu``, we map the solutions from the training set and construct a reduced basis.

```@example 1
T★u_train = [pushfwd(u_train[i], ψᶜ_train[i], fe_spaces.V, dΩ) for i in eachindex(u_train)]
ϕ, evd_T★u = pod(T★u_train, fe_spaces.V, fe_spaces.U, dΩ, τ)
nₘ = length(ϕ)
```

The mapped snapshots are much easier to compress using linear methods as indicated by the decay of the correlation matrix eigenvalues.

```@setup 1
cpal = palette(:thermal, 4);
evds = plot(abs.(evd_u.values) ./ evd_u.values[1], yaxis=:log, minorgrid=true, xaxis=:log,
    yticks=10.0 .^ (-16:2:0), xticks=([1, 10, 100], string.([1, 10, 100])),
    linewidth=2, marker=:circle, xlabel=L"n", ylabel=L"\lambda_n / \lambda_1", ylim=(1e-16, 2), label=L"u", color=cpal[1],
    legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, legend=:bottomleft)
plot!(abs.(evd_T★u.values) ./ evd_T★u.values[1], linewidth=2, marker=:square, markersize=3, label=L"u \circ \Phi^{-1}", color=cpal[2])
plot!(abs.(evd_ψᶜ.values) ./ evd_ψᶜ.values[1], linewidth=2, marker=:diamond, label=L"\nabla \psi^c", color=cpal[3])
```
```@example 1
plot(evds)
```

### Empirical interpolation

To construct the empirical iterpolation, we collect the parameter-dependent forms from the training set and perform a proper orthogonal decomposition. The interpolation functions and points are then obtained starting from these POD modes. 

```@example 1
f_nomap = [interpolate_everywhere(x -> f(x, μ .+ 0.5), fe_spaces.W) for μ in μ_train] # hide
Ξ_f, evd_f = pod(f_nomap, fe_spaces.W, fe_spaces.W, dΩ, τ_eim) # hide
f★J = [get_f★J(f, μ, get_transport_potential(μ, ξᶜ, fe_spaces.Ψ, gp), fe_spaces.W) for μ in μ_train]
Ξ_f★J, evd_f★J = pod(f★J, fe_spaces.W, fe_spaces.W, dΩ, τ_eim)
eim_f★J = EmpiricalInterpolation(Ξ_f★J, (ϕ,X) -> form_f★J(ϕ,X,dΩ), ϕ, fe_spaces.W)
K = [get_K(get_transport_potential(μ, ξᶜ, fe_spaces.Ψ, gp), fe_spaces.W_matrix) for μ in μ_train];
Ξ_K, evd_K = pod(K, fe_spaces.W_matrix, fe_spaces.W_matrix, dΩ, τ_eim)
eim_K = EmpiricalInterpolation(Ξ_K, (ϕ,X) -> form_K(ϕ,X,dΩ), ϕ, fe_spaces.W_matrix)
get_Q(eim_K), get_Q(eim_f★J)
```

```@setup 1
cpal = palette(:thermal, 4);
eim_evds = plot(abs.(evd_f.values) ./ evd_f.values[1], yaxis=:log, minorgrid=true, xaxis=:log,
    yticks=10.0 .^ (-16:2:0), xticks=([1, 10, 100], string.([1, 10, 100])),
    linewidth=2, marker=:circle, xlabel=L"n", ylabel=L"\lambda_n / \lambda_1", ylim=(1e-16, 2), label=L"f", color=cpal[1],
    legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, legend=:bottomleft)
plot!(abs.(evd_f★J.values) ./ evd_f★J.values[1], linewidth=2, marker=:square, markersize=3, label=L"f \circ \Phi^{-1} \det D\Phi^{-1}", color=cpal[2])
plot!(abs.(evd_K.values) ./ evd_K.values[1], linewidth=2, marker=:diamond, label=L"[D\Phi^{-1}]^{-1} [D\Phi^{-1}]^{-T} \det D\Phi^{-1}", color=cpal[3])
```
```@example 1
plot(eim_evds)
```

### Online phase

The online phase consists of evaluating the ``\mu \mapsto w(\mu)_j`` maps and solving the mapped problem in the reference reduced basis. Lastly, the solution is mapped back for plotting using the c-transform.

```@example 1
ψᶜ_test = [get_transport_potential(μ, ξᶜ, fe_spaces.Ψ, gp) for μ in μ_test]
ũE_trb_eim = [snapshot(f, μ_test[i], fe_spaces.V, fe_spaces.U, dΩ, ϕ, ψᶜ_test[i], eim_f★J, eim_K, P) for i in eachindex(μ_test)];
nothing # hide
```

```@example 1
T★u_trb_eim = [FEFunction(fe_spaces.V, uE[1]' * get_free_dof_values.(ϕ)) for uE in ũE_trb_eim]
E_trb_eim = [uE[2] for uE in ũE_trb_eim]
ψ̂_test = [c_transform(ψᶜ, fe_spaces.V_fine, c, log_ρ̂_ref, MC, ε_fine) for ψᶜ in ψᶜ_test]
ψ_test = [boundary_projection(ψ, δ⁻¹, κ, fe_spaces.Ψ, dΩ, 2 * highorder) for ψ in ψ̂_test]
u_trb_eim = [pushfwd(T★u_trb_eim[i], ψ_test[i], fe_spaces.V, dΩ) for i in eachindex(μ_test)]
ΔL2_trb_eim = rel_error_vec(u_trb_eim, u_test, L2, dΩ)
@printf "registered (n = %.0f, m = %.0f) \t L2 error avg.: %.2e ± %.2e \t max.: %.2e \n" nₘ m Statistics.mean(ΔL2_trb_eim) Statistics.std(ΔL2_trb_eim) maximum(ΔL2_trb_eim)
```

Lastly, we plot some cross-sections of the worst approximation.

```@setup 1
_i = argmax(ΔL2_trb_eim)
μ = μ_test[_i]
cpal = palette(:thermal, 5)
cross_sec_1 = plot(0:0.005:1, x -> u_test[_i](Point(x, 0.5 + μ_test[_i][2])), linewidth=2, color=cpal[1], xlabel=L"x",
                label=L"u", legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, dpi=400)
plot!(0:0.005:1, x -> u_rb[_i](Point(x, 0.5 + μ_test[_i][2])), linewidth=2, color=cpal[2], label=L"u_{\mathrm{rb}}")
plot!(0:0.005:1, x -> u_trb_eim[_i](Point(x, 0.5 + μ_test[_i][2])), linewidth=2, color=cpal[4], label=L"u_{\mathrm{trb, eim}}")
cross_sec_2 = plot(0:0.005:1, x -> u_test[_i](Point(0.5 + μ_test[_i][1], x)), linewidth=2, color=cpal[1], xlabel=L"y",
                label=L"u", legendfontsize=12, tickfontsize=8, xguidefontsize=12, yguidefontsize=12, dpi=400)
plot!(0:0.005:1, x -> u_rb[_i](Point(0.5 + μ_test[_i][1], x)), linewidth=2, color=cpal[2], label=L"u_{\mathrm{rb}}")
plot!(0:0.005:1, x -> u_trb_eim[_i](Point(0.5 + μ_test[_i][1], x)), linewidth=2, color=cpal[4], label=L"u_{\mathrm{trb, eim}}")
```

```@example 1
plot(cross_sec_1, cross_sec_2)
```

