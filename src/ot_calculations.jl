function reference_density(ρ, uₕ, nᵦ, MC)
    N_fine = MC.n - 1
    ūₕ = [ reshape( (get_free_dof_values( interpolate_everywhere(Interpolable(ρ(_uₕ)), V₁) )),N_fine+1,N_fine+1) for _uₕ in uₕ]
    for _ūₕ in ūₕ
        _ūₕ ./= sum(_ūₕ) * (1 / N_fine)^2
    end

    SPB = SinkhornParameters(256, ε)
    SPB.tol = 1e-12
    SPB.debias = true
    SPB.averaged_updates = false
    SPB.update_potentials = false

    ūₕᵦ = ūₕ[rand(1:length(uₕ),nᵦ)]
    ūₕ_ref = sinkhorn_barycenter_sep([ 1/nᵦ for _ in ūₕᵦ], ūₕᵦ, k, SPB, MC)

    # transport potentials
    SP = SinkhornParameters(128, ε)
    SP.tol = 1e-12
    SP.debias = false
    SP.averaged_updates = false
    SP.update_potentials = true

    ψ̄ᶜ = get_ψ̄ᶜ(ūₕ, ūₕ_ref, SP, MC)

    return ūₕ_ref, ψ̄ᶜ
end

function get_ρ̂(u, ρ::Function, V, N)
    ρ̂ = reshape( get_free_dof_values( interpolate_everywhere(Interpolable(ρ(u)), V) ), N, N)
    ρ̂ ./= sum(ρ̂)
    return ρ̂
end

function get_ρ̂(f::Function, V, N)
    ρ̂ = reshape( get_free_dof_values( interpolate_everywhere( f, V) ), N, N)
    ρ̂ ./= sum(ρ̂)
    return ρ̂
end

function get_ψ̂_ψ̂ᶜ(ρ̂, ρ̂_ref, k, SP, MC)
    a₀ = zero(ρ̂)
    b₀ = zero(ρ̂)
    dα₀ = zero(ρ̂)
    dβ₀ = zero(ρ̂)
    a₀ .= 1.0
    b₀ .= 1.0
    dα₀ .= 1.0
    dβ₀ .= 1.0
    sinkhorn_dvg_sep(ρ̂, ρ̂_ref, a₀, b₀, dα₀, dβ₀, k, SP, MC)
    return (log.(a₀) - log.(dα₀) ) * SP.ε, (log.(b₀) - log.(dβ₀) ) * SP.ε
end

function c_transform(ψᶜ, V₁, c, log_ρ̂_ref, MC, ε_fine)
    N_fine = size(log_ρ̂_ref,1) - 1

    ψᶜ₁ = interpolate_everywhere(Interpolable(ψᶜ), V₁)
    ψ̄ᶜ = reshape( get_free_dof_values(ψᶜ₁), N_fine+1, N_fine+1)
    ψ̄ = zero(ψ̄ᶜ)
    WassersteinDictionaries.softmin_separated!(ψ̄, ψ̄ᶜ, log_ρ̂_ref, ε_fine, c, MC[:t1,Float64], MC[:t2,Float64])

    ψ̄
end

function safe_log(x)
    if x > 1e-20
        return log(x)
    else
        return -100.0
    end
end
