function trainparameterset(nₛ, μ_max, μ_min)
    μ̄ = []
    for i in 1:nₛ
        if i == 1
            μ = zeros(2)
        else
            μ = rand(2) .* (μ_max - μ_min) .+ μ_min
        end
        push!(μ̄, μ)
    end
    return μ̄
end

function testparameterset(nₛ, μ_max, μ_min)
    μ̄ = []
    for i in 1:nₛ
        μ = rand(2) .* (μ_max - μ_min) .+ μ_min
        push!(μ̄, μ)
    end
    return μ̄
end

function snapshots(f, μ̄, V₂, U₂, dΩ)
    uₕ = []

    for i in eachindex(μ̄)
        μ = μ̄[i]
        x0 = μ .+ 0.5
        _f(x) = f(x, x0)
        #push!(κₕ, interpolate_everywhere(_f, V₂))

        a(u,v) = ∫( (∇(v)⋅∇(u)) )dΩ
        b(v) = ∫( _f*v )dΩ

        op = AffineFEOperator(a,b,U₂,V₂)
        push!(uₕ, solve(op))
    end
    return uₕ
end

function snapshots_rb(f, μ̄, ϕ, V₂, U₂, dΩ)
    uₕ_rb = []
    Aᵣ = [ sum(∫( (∇(ϕ[i])⋅∇(ϕ[j])) )dΩ) for i in eachindex(ϕ), j in eachindex(ϕ) ] # reduced Laplace operator
    for i in eachindex(μ̄ₜ)
        μ = μ̄[i]
        x0 = μ .+ 0.5
        _f(x) = f(x, x0)

        bᵣ = [ sum(∫( ϕ[i]*_f )dΩ) for i in eachindex(ϕ) ]
        _u = Aᵣ \ bᵣ
        _uₕ = FEFunction(V₂, _u' * get_free_dof_values.(ϕ[1:n]) )

        push!(uₕ_rb, _uₕ)
    end
    return uₕ_rb
end

function snapshots_trb(f, μ̄ₜ, ϕσ, V₂, U₂, dΩ, domain, ξᶜ, gp, Ψ, V₁, N_fine, log_ūₕ_ref, MC)
    uₕₜ_trb_ref = []
    uₕₜ_trb = []
    ξ̂ᶜ = get_free_dof_values.(ξᶜ)
    id = TensorValue(diagm(ones(2)))
    for _i in eachindex(μ̄ₜ)
        μ = μ̄ₜ[_i]
        x0 = μ .+ 0.5
        _f(x) = f(x, x0)
        _λ = get_λ(μ, gp)
        _ψᶜ = get_ψᶜ(_λ, ξ̂ᶜ, Ψ)
        psi_cache = Gridap.CellData.return_cache(∇(_ψᶜ), Point(0,0))
        # right hand side
        function f_σ(y)
            dy = Gridap.evaluate!(psi_cache, ∇(_ψᶜ), y)
            Ty = y - dy
            return _f(Ty)
        end
        fσₕ = interpolate_everywhere(f_σ, V₂);

        DT = id - ∇∇(_ψᶜ)
        J = abs(det(DT))

        Aᵣ = [ sum(∫((inv(DT)⋅∇(ϕσ[j]))⋅(inv(DT)⋅∇(ϕσ[i])) * J )dΩ) for i in eachindex(ϕσ), j in eachindex(ϕσ) ]
        bᵣ = [ sum(∫( ϕσ[i] * fσₕ * J )dΩ) for i in eachindex(ϕσ) ]
        _u = Aᵣ \ bᵣ

        _uₕ = FEFunction(V₂, _u' * get_free_dof_values.(ϕσ) )
        _ψ = c̄_transform(_λ, ξᶜ, Ψ, V₁, N_fine, log_ūₕ_ref, MC)
        _uₕₜ_trb = pushfwd(_uₕ, _ψ, V₂, domain, dΩ)

        push!(uₕₜ_trb_ref, _uₕ)
        push!(uₕₜ_trb, _uₕₜ_trb)
    end
    return uₕₜ_trb, uₕₜ_trb_ref
end