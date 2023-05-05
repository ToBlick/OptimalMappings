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

# obtain n from retained ev energy
function get_n(v, tol = 1e-3)
    for i in 1:length(v)
        if sum(v[1:i])/sum(v) > 1 - tol
            return i
        end
    end
end

# obtain reduced basis from snapshots uₕ and evd of correlation matrix
function get_ϕ(n, uₕ, evd, V₂, U₂)
    ϕ = []
    for i in 1:n
        fv = zero_free_values(U₂) # dof vector
        for j in eachindex(uₕ)
            fv_j = get_free_dof_values(uₕ[j])
            fv .+= fv_j * evd.vectors[j,i]
        end
        fv ./= sqrt(evd.values[i])
        _ϕ = FEFunction(V₂, fv)
        push!(ϕ, _ϕ)
    end
    return ϕ
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

function safe_log(x)
    if x > 1e-20
        return log(x)
    else
        return -100.0
    end
end

function get_ψ̄ᶜ(ūₕ, ūₕ_ref, SP, MC)
    # compute transport potentials from ū to ū_ref
    ψ̄ᶜ = [ zero(_ūₕ) for _ūₕ in ūₕ ] 
    a₀ = zero(ūₕ[1])
    b₀ = zero(ūₕ[1])
    d₀₁ = zero(ūₕ[1])
    d₀₂ = zero(ūₕ[1])
    for s in eachindex(ūₕ)

        a₀ .= 1.0
        b₀ .= 1.0


        sinkhorn_dvg_sep( ūₕ[s], ūₕ_ref, a₀, b₀, d₀₁, d₀₂, k, SP, MC)
        ψ̄ᶜ[s] .= log.(b₀) * ε
    end
    return ψ̄ᶜ
end

function boundary_projection(ψ̄ᶜ, N_fine, model, degree, V₁, Ψ, dΩ)
    ψᶜ = []
    Γ = BoundaryTriangulation(model)
    nb = get_normal_vector(Γ)
    dΓ = Measure(Γ,degree)
    δ = 1e9 * N_fine
    for i in eachindex(ψ̄ᶜ)
        ψ₁ = interpolate_everywhere( Interpolable( FEFunction(V₁, vec(ψ̄ᶜ[i]))), Ψ)
        a(u,v) = ∫( (∇(v)⋅∇(u)) )dΩ + ∫( v*u )dΩ + δ * ∫( (nb⋅∇(u))*v )dΓ
        b(v) = ∫( (∇(v)⋅∇(ψ₁)) )dΩ + ∫( v*ψ₁ )dΩ
        op = AffineFEOperator(a,b,TrialFESpace(Ψ),Ψ)
        push!(ψᶜ, solve(op))
    end
    return ψᶜ
end

function c̄_transform(_λ, ξᶜ, Ψ, V₁, N_fine, log_ūₕ_ref, MC)
    _ψᶜᵣ = FEFunction(Ψ, _λ' * get_free_dof_values.(ξᶜ))
    _ψᶜᵣ₀ = interpolate_everywhere(Interpolable(_ψᶜᵣ), V₁)
    _ψ̄ᶜᵣ = reshape( get_free_dof_values(_ψᶜᵣ₀), N_fine+1, N_fine+1)
    _ψ̄ᵣ = zero(_ψ̄ᶜᵣ)
    WassersteinDictionaries.softmin_separated!(_ψ̄ᵣ, _ψ̄ᶜᵣ, log_ūₕ_ref, 1e-6, c, MC[:t1,Float64], MC[:t2,Float64])
    _ψᵣ₀ = FEFunction(V₁, vec(_ψ̄ᵣ))
    _ψᵣ = interpolate_everywhere( Interpolable(_ψᵣ₀), Ψ)
    _ψᵣ
end

function pushfwd(u,ψ,V,domain,dΩ)
    T = typeof(u(Point(0,0)))
    u_cache = Gridap.CellData.return_cache(u, Point(0,0))
    psi_cache = Gridap.CellData.return_cache(∇(ψ), Point(0,0))
    function uT(y)
        dy = Gridap.evaluate!(psi_cache, ∇(ψ), y)
        Ty = y - dy
        if Ty[1] < domain[1] || Ty[1] > domain[2] || Ty[2] < domain[3] || Ty[2] > domain[4]
            return zero(T)
        else
            return Gridap.evaluate!(u_cache, u, Ty)
        end
    end
    U = u.fe_space
    a(u,v) = ∫( u*v )*dΩ
    l(v) = ∫( v*uT )*dΩ
    op = AffineFEOperator(a,l,U,V)
    return solve(op)
end

function get_gp(μ̄, λ, m)
    mZero = MeanZero()
    kern = SE(0.0,0.0)
    logObsNoise = -6.0 

    gp = []
    _μ = [ μ̄[i][k] for k in 1:d, i in eachindex(μ̄)]
    for _m in 1:m
        _λ = [_λ[_m] for _λ in λ]

        _gp = GP(_μ, _λ, mZero, kern, logObsNoise)
        push!(gp, _gp)
    end
    return gp
end

function get_λ(μ, gp)
    _μ = zeros(2,1)
    _μ .= μ
    [ predict_y(_gp, _μ)[1][1] for _gp in gp ]
end

function get_ψᶜ(λ, ξ̂ᶜ, Ψ)
    FEFunction(Ψ, λ' * ξ̂ᶜ)
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