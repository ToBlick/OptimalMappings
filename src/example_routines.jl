"Return `nₛ` 2d samples between `μ_min` and `μ_max`"
function testparameterset(nₛ, μ_max, μ_min)
    μ̄ = []
    for i in 1:nₛ
        μ = rand(2) .* (μ_max - μ_min) .+ μ_min
        push!(μ̄, μ)
    end
    return μ̄
end

### for example 1

function snapshot(f, μ, V₂, U₂, dΩ, ::PoissonProblem)
    x0 = μ .+ 0.5
    _f(x) = f(x, x0)

    a(u,v) = ∫( (∇(v)⋅∇(u)) )dΩ
    b(v) = ∫( _f*v )dΩ

    op = AffineFEOperator(a,b,U₂,V₂)
    u = solve(op)
    return u, sum(a(u,u)) / 2
end

function get_b(f, μ, ϕ::Vector{<:FEFunction}, dΩ)
    x0 = μ .+ 0.5
    _f(x) = f(x, x0)
    return [ sum(∫( ϕ[i]*_f )dΩ) for i in eachindex(ϕ) ]
end

function get_A(μ, ϕ::Vector{<:FEFunction}, dΩ)
    return [ sum(∫( (∇(ϕ[i])⋅∇(ϕ[j])) )dΩ) for i in eachindex(ϕ), j in eachindex(ϕ) ] 
end

function snapshot(f, μ, V₂, U₂, dΩ, ϕ::Vector{<:FEFunction}, ::PoissonProblem)
    A = get_A(μ, ϕ, dΩ)
    b = get_b(f, μ, ϕ, dΩ)
    u = A \ b
    return u, u' * A * u / 2
end

function snapshot(f, μ, V₂, U₂, dΩ, ϕ::Vector{<:FEFunction}, A::Matrix, ::PoissonProblem)
    b = get_b(f, μ, ϕ, dΩ)
    u = A \ b
    return u, u' * A * u / 2
end

### mapped RB

function snapshot(f, μ, V₂, U₂, dΩ, ϕσ::Vector{<:FEFunction}, ψᶜ::FEFunction, ::PoissonProblem)
    x0 = μ .+ 0.5
    _f(x) = f(x, x0)
    id = TensorValue(diagm(ones(2)))
    DT = id - ∇∇(ψᶜ)
    J = abs(det(DT))
    psi_cache = Gridap.CellData.return_cache(∇(ψᶜ), Point(μ...))
    function T(y::T) where T
        dy = Gridap.evaluate!(psi_cache, ∇(ψᶜ), y)
        return (y - dy)::T
    end
    T★f = interpolate_everywhere(_f ∘ T, V₂);

    A = [ sum(∫((inv(DT)⋅∇(ϕσ[j]))⋅(inv(DT)⋅∇(ϕσ[i])) * J )dΩ) for i in eachindex(ϕσ), j in eachindex(ϕσ) ]
    b = [ sum(∫( ϕσ[i] * T★f * J )dΩ) for i in eachindex(ϕσ) ]
    ũ = A \ b

    return ũ, ũ' * A * ũ / 2
end

### mapped deim RB

function snapshot(f, μ, V₂::FESpace, U₂::FESpace, dΩ, ϕσ::Vector{<:FEFunction}, ψᶜ::FEFunction, 
            eim_f★J::EmpiricalInterpolation, eim_K★J::EmpiricalInterpolation, ::PoissonProblem)
    
    nₘ = length(ϕσ)
    
    x0 = μ .+ 0.5
    _f(x) = f(x, x0)
    
    id = TensorValue(diagm(ones(2)))
    DT = id - ∇∇(ψᶜ)
    J = abs(det(DT))

    psi_cache = Gridap.CellData.return_cache(∇(ψᶜ), Point(μ...))
    function T(y::T) where T
        dy = Gridap.evaluate!(psi_cache, ∇(ψᶜ), y)
        return (y - dy)::T
    end
    T★f(y) = _f(T(y))

    f_vals = Gridap.FESpaces._cell_vals(eim_f★J.W, T★f * J) # lazy array
    f_cache = array_cache(f_vals)
    K_vals = Gridap.FESpaces._cell_vals(eim_K★J.W, inv(DT) ⋅ inv(DT) * J)
    K_cache = array_cache(K_vals)

    B_f = get_interpolation_matrix(eim_f★J)
    A_f = get_precomputed_matrix(eim_f★J)
    idxs_f = get_indices(eim_f★J)
    β_f = B_f \ [ getindex!(f_cache, f_vals, i[1])[i[2]] for i in idxs_f ]
    b =  (β_f' * A_f)'

    B_K = get_interpolation_matrix(eim_K★J)
    A_K = get_precomputed_matrix(eim_K★J)
    idxs_K = get_indices(eim_K★J)
    β_K = B_K \ [ getindex!(K_cache, K_vals, i[1])[i[2]] for i in idxs_K ]

    A = Matrix(reshape(β_K' * A_K, nₘ, nₘ))

    ũ = A \ b

    return ũ, ũ' * A * ũ / 2
end

### for example 2

function snapshots(u₀, ω_samples::Vector{T}, V₂, U₂, dΩ, nls, θ, t0, dt, tF, P::NonlinearAdvectionProblem) where {T}
    
    u₀ₕ = interpolate_everywhere(u₀, U₂(0.0))

    u = typeof(u₀ₕ)[] 
    ω = T[]
    t = T[]

    for _ω in ω_samples
            _w̄(x) = P.w̄(x, _ω)
            push!(u, FEFunction(V₂, copy(get_free_dof_values(u₀ₕ))))
            push!(ω, _ω)
            push!(t, t0)
            res(t,u,v) = ∫( ∂t(u)*v + 
                            + P.β * (∇(v)⋅∇(u)) 
                            - (P.γ .* u .+ 1) .* (∇(v) ⋅ _w̄) .* u
                            )dΩ # residual for NL solver
            
            op = TransientFEOperator(res,U₂,V₂)
            ode_solver = ThetaMethod(nls,dt,θ)
            sol_t = solve(ode_solver,op,u₀ₕ,t0,tF)

            for (_u, _t) in sol_t
                if abs(_t % 0.05) < 1e-6 || abs(_t % 0.05 - 0.05) < 1e-6
                    push!(u, FEFunction(V₂, copy(get_free_dof_values(_u))))
                    push!(ω, _ω)
                    push!(t, _t)
                end
            end
    end
    return u, ω, t
end


function snapshots(nu₀, ω_samples, ϕ::Vector{<:FEFunction}, V₂, U₂, dΩ, nls, θ, t0, dt, tF, P::NonlinearAdvectionProblem)

    # RB projection
    ũ₀ = [ sum(∫(nu₀ * _ϕ)dΩ) for _ϕ in ϕ]
    u₀ = FEFunction(V₂, copy(ũ₀)' * get_free_dof_values.(ϕ))
    u = typeof(u₀)[] 

    n = length(ϕ)

    for _ω in ω_samples
        _w̄(x) = P.w̄(x, _ω)

        # Assemble reduced operators
        _K = (P.β * [ sum(∫( ∇(ϕ[i]) ⋅ ∇(ϕ[j]) )dΩ) for i in 1:n, j in 1:n ]
                - [ sum(∫((∇(ϕ[i]) ⋅ _w̄) .* ϕ[j])dΩ) for i in 1:n, j in 1:n ] )
            
        _A = zeros(n,n,n)
        if P.γ == 0
            nothing
        else
            _A .= - P.γ * [ sum(∫(∇(ϕ[i]) ⋅ _w̄ * ϕ[j] * ϕ[k])dΩ) for i in 1:n, j in 1:n, k in 1:n ]
        end

        _M = diagm(ones(n))

        M(t) = _M
        K(t) = _K
        A(t) = _A

        push!(u, u₀)
        ode_solver = ThetaMethod(nls,dt,θ)
        rbop = RBODEOperator(M, K, A)

        rbsol_t = solve(ode_solver, rbop, ũ₀, t0, tF)

        for (ũ, t) in rbsol_t
            if abs(t % 0.05) < 1e-6 || abs(t % 0.05 - 0.05) < 1e-6
                _u = FEFunction(V₂, copy(ũ)' * get_free_dof_values.(ϕ))
                push!(u, _u)
            end
        end
    end
    return u
end

function snapshots(nu₀, ω_samples, ϕ::Vector{<:FEFunction}, V₂, U₂, dΩ, nls, θ, t0, dt, tF, eim_ā, eim_āu, P::NonlinearAdvectionProblem)

    # RB projection
    ũ₀ = [ sum(∫(nu₀ * _ϕ)dΩ) for _ϕ in ϕ]
    u₀ = FEFunction(V₂, copy(ũ₀)' * get_free_dof_values.(ϕ))
    u = typeof(u₀)[] 

    n = length(ϕ)

    _K = P.β * [ sum(∫( ∇(ϕ[i]) ⋅ ∇(ϕ[j]) )dΩ) for i in 1:n, j in 1:n ]
    _M = diagm(ones(n))

    for _ω in ω_samples
        _w̄(x) = P.w̄(x, _ω)

        a_vals = Gridap.FESpaces._cell_vals(eim_ā.W, _w̄)
        a_cache = array_cache(a_vals)
        B_a = get_interpolation_matrix(eim_ā)
        A_a = get_precomputed_matrix(eim_ā)
        idxs_a = get_indices(eim_ā)

        A_au = get_precomputed_matrix(eim_āu)

        θ_a = B_a \ [ getindex!(a_cache, a_vals, i[1])[i[2]] for i in idxs_a ]
        K_a = _K - Matrix(reshape(θ_a' * A_a, n, n))

        _A = - P.γ * reshape(θ_a' * A_au, n, n, n)

        M(t) = _M
        K(t) = K_a
        A(t) = _A

        push!(u, u₀)
        ode_solver = ThetaMethod(nls,dt,θ)
        rbop = RBODEOperator(M, K, A)

        rbsol_t = solve(ode_solver, rbop, ũ₀, t0, tF)

        for (ũ, t) in rbsol_t
            if abs(t % 0.05) < 1e-6 || abs(t % 0.05 - 0.05) < 1e-6
                _u = FEFunction(V₂, copy(ũ)' * get_free_dof_values.(ϕ))
                push!(u, _u)
            end
        end
    end
    return u
end

function snapshots(nu₀, ω_samples, ϕ, V₂, U₂, dΩ, ξᶜ, gp, Ψ, nls, θ, t0, dt, tF, 
                eim_J, eim_∂tΦ★J, eim_ā★J, eim_K, eim_āu★J, P::NonlinearAdvectionProblem)

    ξ̂ᶜ = get_free_dof_values.(ξᶜ)

    id = TensorValue(diagm(ones(2)))
    ### initial condition
    λ = get_λ( [t0, 0.0], gp)
    ψᶜ = FEFunction(Ψ, λ' * ξ̂ᶜ)
    ∂tψᶜ = FEFunction(Ψ, λ' * ξ̂ᶜ)
    DT = id - ∇∇(ψᶜ)
    J = abs(det(DT))

    Tu₀(x) = nu₀(x - ∇(ψᶜ)(x))
    ũ₀ = [ sum(∫(_ϕ)dΩ) for _ϕ in ϕ]
    ũ = typeof(ũ₀)[]

    nₘ = length(ϕ)

    J_vals = Gridap.FESpaces._cell_vals(eim_J.W, J) # lazy array
    J_cache = array_cache(J_vals)
    B_J = get_interpolation_matrix(eim_J)
    A_J = get_precomputed_matrix(eim_J)
    idxs_J = get_indices(eim_J)

    K_vals = Gridap.FESpaces._cell_vals(eim_K.W, inv(DT) ⋅ inv(DT) * J)
    K_cache = array_cache(K_vals)
    B_K = get_interpolation_matrix(eim_K)
    A_K = get_precomputed_matrix(eim_K)
    idxs_K = get_indices(eim_K)

    ∂tΦ★J_vals = Gridap.FESpaces._cell_vals(eim_∂tΦ★J.W, inv(DT) ⋅ ∇(∂tψᶜ) * J)
    ∂tΦ★J_cache = array_cache(∂tΦ★J_vals)
    B_∂tΦ★J = get_interpolation_matrix(eim_∂tΦ★J)
    A_∂tΦ★J = get_precomputed_matrix(eim_∂tΦ★J)
    idxs_∂tΦ★J = get_indices(eim_∂tΦ★J)

    @time for _ω in ω_samples
        
        _w̄(x) = P.w̄(x, _ω)
        
        ā★J_vals = Gridap.FESpaces._cell_vals(eim_ā★J.W, inv(DT) ⋅ _w̄ * J)
        ā★J_cache = array_cache(ā★J_vals)
        B_ā★J = get_interpolation_matrix(eim_ā★J)
        A_ā★J = get_precomputed_matrix(eim_ā★J)
        idxs_ā★J = get_indices(eim_ā★J)
    
        A_āu★J = get_precomputed_matrix(eim_āu★J)

        function M(t)
        
            λ = get_λ( [t, _ω], gp)
            get_free_dof_values(ψᶜ) .= λ' * ξ̂ᶜ

            θ_J = B_J \ [ getindex!(J_cache, J_vals, i[1])[i[2]] for i in idxs_J ]
            M = Matrix(reshape(θ_J' * A_J, nₘ, nₘ))
        
            return M
        end

        function K(t)
            
            λ = get_λ( [t, _ω], gp)
            get_free_dof_values(ψᶜ) .= λ' * ξ̂ᶜ
        
            ∂tλ = ( get_λ( [t + dt/2, _ω], gp) - get_λ( [t - dt/2, _ω], gp) ) / dt
            get_free_dof_values(∂tψᶜ) .= ∂tλ' * ξ̂ᶜ

            β_K = B_K \ [ getindex!(K_cache, K_vals, i[1])[i[2]] for i in idxs_K ]
            K = Matrix(reshape(β_K' * A_K, nₘ, nₘ))

            β_∂tΦ★J = B_∂tΦ★J \ [ getindex!(∂tΦ★J_cache, ∂tΦ★J_vals, i[1])[i[2]] for i in idxs_∂tΦ★J ]
            A1 = Matrix(reshape(β_∂tΦ★J' * A_∂tΦ★J, nₘ, nₘ))

            β_ā★J = B_ā★J \ [ getindex!(ā★J_cache, ā★J_vals, i[1])[i[2]] for i in idxs_ā★J ]
            A2 = Matrix(reshape(β_ā★J' * A_ā★J, nₘ, nₘ))

            return P.β * K + A1 - A2
        end

        function A(t)
                λ = get_λ( [t, _ω], gp)
                get_free_dof_values(ψᶜ) .= λ' * ξ̂ᶜ

                β_ā★J = B_ā★J \ [ getindex!(ā★J_cache, ā★J_vals, i[1])[i[2]] for i in idxs_ā★J ]
                return - P.γ * reshape(β_ā★J' * A_āu★J, nₘ, nₘ, nₘ)
        end

        trbop = RBODEOperator(M, K, A)

        ũ₀ .= trbop.M(t0) \ [ sum(∫( Tu₀ * _ϕ * J)dΩ) for _ϕ in ϕ]

        trb_ode_solver = ThetaMethod(nls, dt, θ)
        trbsol_t = solve(trb_ode_solver, trbop, ũ₀, t0, tF)

        push!(ũ, copy(ũ₀))

        for (u,t) in trbsol_t
            if abs(t % 0.05) < 1e-6 || abs(t % 0.05 - 0.05) < 1e-6
                push!(ũ, copy(u))
            end
        end
        #println("ω = $_ω")
    end
    return ũ
end