import Gridap.ODEs.ODETools: ODEOperator
import Gridap.ODEs.ODETools: Nonlinear
import Gridap.ODEs.ODETools: allocate_cache
import Gridap.ODEs.ODETools: update_cache!
import Gridap.ODEs.ODETools: allocate_residual
import Gridap.ODEs.ODETools: jacobian!
import Gridap.ODEs.ODETools: jacobians!
import Gridap.ODEs.ODETools: allocate_jacobian
import Gridap.ODEs.ODETools: residual!

struct RBODEOperator{T<:Real,C} <: ODEOperator{C}
    M # mass matrix
    K # linear part
    A # quadratic part
    order::Integer
end
  
get_order(op::RBODEOperator) = op.order
  
function residual!(r::AbstractVector,op::RBODEOperator,t::Real,x::NTuple{2,AbstractVector},ode_cache)
    u,u_t = x
    A_uu = zero(u)
    if μ != 0
        for j in eachindex(u), k in eachindex(u)
            A_uu .+= op.A(t)[:,j,k] * u[j] * u[k] 
        end
    end
    r .= op.M(t) * u_t .+ op.K(t) * u .+ A_uu
end
  
function allocate_residual(op::RBODEOperator,u::AbstractVector,cache)
    zero(u)
end

function jacobian!(J::AbstractMatrix,
    op::RBODEOperator,
    t::Real,
    x::NTuple{2,AbstractVector},
    i::Int,
    γᵢ::Real,
    ode_cache)
    @assert get_order(op) == 1
    @assert 0 < i <= get_order(op)+1
    u,u_t = x

    if i==1
        A_u1 = zero(op.K(t))
        A_u2 = zero(op.K(t))
        if μ != 0
            for i in eachindex(u), j in eachindex(u), k in eachindex(u)
                A_u1[i,j] += op.A(t)[i,j,k] * u[k] # first index is test function, second is trial. k is summed
                A_u2[i,j] += op.A(t)[i,k,j] * u[k]
            end
        end
        J .+= op.K(t) .+ A_u1 .+ A_u2
    elseif i==2
        J .+= op.M(t) .* γᵢ
    end
    J
end

function jacobians!(
    J::AbstractMatrix,
    op::RBODEOperator,
    t::Real,
    x::Tuple{Vararg{AbstractVector}},
    γ::Tuple{Vararg{Real}},
    ode_cache)
    @assert length(γ) == get_order(op) + 1
    for order in 1:get_order(op)+1
      jacobian!(J,op,t,x,order,γ[order],ode_cache)
    end
    J
  end
  
function allocate_jacobian(op::RBODEOperator,u::AbstractVector,cache)
    spzeros(size(op.M(0.0)))
end
  
allocate_cache(op::RBODEOperator) = nothing
allocate_cache(op::RBODEOperator,v::AbstractVector) = (similar(v),nothing)
allocate_cache(op::RBODEOperator,v::AbstractVector,a::AbstractVector) = (similar(v),similar(a),nothing)
update_cache!(cache,op::RBODEOperator,t::Real) = cache
  
function snapshots(nu₀, ω̄_samples, V₂, U₂, dΩ)
    uₕ = [ ]
    ω̄ = [ ]
    t̄ = [ ]

    u₀ = interpolate_everywhere(nu₀, U₂(0.0))

    for _ω in ω̄_samples
            _w̄(x) = w̄(x, _ω)
            push!(uₕ, FEFunction(V₂, copy(get_free_dof_values(u₀))))
            push!(ω̄, _ω)
            push!(t̄, t0)
            res(t,u,v) = ∫( ∂t(u)*v + 
                            + β * (∇(v)⋅∇(u)) 
                            - (μ .* u .+ 1) .* (∇(v) ⋅ _w̄) .* u
                            )dΩ # residual for NL solver
            
            op = TransientFEOperator(res,U₂,V₂)
            sol_t = solve(ode_solver,op,u₀,t0,tF)

            for (_uₕ, _t) in sol_t
                push!(uₕ, FEFunction(V₂, copy(get_free_dof_values(_uₕ))))
                push!(ω̄, _ω)
                push!(t̄, _t)
            end
    end
    return uₕ, ω̄, t̄
end

function snapshots_rb(nu₀, ω̄ₜ_samples, ϕ, V₂, U₂, dΩ)
    uₕ = [ ]
    #ω̄ = [ ]
    #t̄ = [ ]

    # RB projection
    ũ₀ = [ sum(∫(nu₀ * _ϕ)dΩ) for _ϕ in ϕ] # nu₀ vs u₀ !!!

    for _ω in ω̄ₜ_samples
        _w̄(x) = w̄(x, _ω)

        # Assemble reduced operators
        _K = (β * [ sum(∫( ∇(ϕ[i]) ⋅ ∇(ϕ[j]) )dΩ) for i in 1:n, j in 1:n ]
                - [ sum(∫((∇(ϕ[i]) ⋅ _w̄) .* ϕ[j])dΩ) for i in 1:n, j in 1:n ] )

        if μ == 0
            _A = zeros(n,n,n)
        else
            _A = - μ * [ sum(∫(∇(ϕ[i]) ⋅ _w̄ * ϕ[j] * ϕ[k])dΩ) for i in 1:n, j in 1:n, k in 1:n ]
        end
        
        _M = diagm(ones(n))

        push!(uₕ, FEFunction(V₂, copy(ũ₀)' * get_free_dof_values.(ϕ)))
        #push!(ω̄, _ω)
        #push!(t̄, t0)

        rbop = RBODEOperator{Float64,Nonlinear}(t -> _M, t -> _K, t -> _A, 1)
        rbsol_t = solve(ode_solver, rbop, ũ₀, t0, tF)

        for (u, t) in rbsol_t
            _uₕₜ_rb = FEFunction(V₂, copy(u)' * get_free_dof_values.(ϕ))
            push!(uₕ, _uₕₜ_rb)
            #push!(ω̄, _ω)
            #push!(t̄, t)
        end
    end
    return uₕ #, ω̄, t̄
end

function snapshots_trb(nu₀, ω̄ₜ_samples, ϕσ, V₂, U₂, dΩ, domain, ξᶜ, gp, Ψ, V₁, N_fine, log_ūₕ_ref, MC)

    uₕₜ_trb_ref = []
    uₕₜ_trb = []
    ξ̂ᶜ = get_free_dof_values.(ξᶜ)
    id = TensorValue(diagm(ones(2)))
    ξ̂ᶜ = get_free_dof_values.(ξᶜ)

    for _ω in ω̄ₜ_samples
        
        _w̄(x) = w̄(x, _ω)

        function M(t)
            M = zeros(nₘ, nₘ)
        
            _λ = get_λ( [t, _ω], gp)
            _ψᶜ = get_ψᶜ(_λ, ξ̂ᶜ, Ψ)
        
            DT = id - ∇∇(_ψᶜ)
            J = abs(det(DT))
            for i in 1:nₘ, j in 1:nₘ
                M[i,j] = sum(∫(ϕσ[i] * ϕσ[j] * J)dΩ)
            end
        
            return M
        end

        function K(t)
            K = zeros(nₘ, nₘ)
            
            _λ = get_λ( [t, _ω], gp)
            _ψᶜ = get_ψᶜ(_λ, ξ̂ᶜ, Ψ)
        
            _∂tλ = ( get_λ( [t + dt/2, _ω], gp) - get_λ( [t - dt/2, _ω], gp) ) / dt
            _∂tψᶜ = get_ψᶜ(_∂tλ, ξ̂ᶜ, Ψ)
        
            DT = id - ∇∇(_ψᶜ)
            J = abs(det(DT))
            
            for i in 1:nₘ, j in 1:nₘ
                K[i,j] = (  β * sum(∫( ∇(ϕσ[i]) ⋅ inv(DT) ⋅ inv(DT) ⋅ ∇(ϕσ[j]) * J )dΩ)
                            - sum(∫( ∇(ϕσ[i]) ⋅ inv(DT) ⋅ _w̄ * ϕσ[j] * J )dΩ)
                            + sum(∫( ϕσ[i] * ∇(ϕσ[j]) ⋅ inv(DT) ⋅ ∇(_∂tψᶜ) * J )dΩ)
                            )
            end
        
            return K
        end

        function A(t)

            A = zeros(nₘ, nₘ, nₘ)
            if μ == 0
                nothing
            else
                _λ = get_λ( [t, _ω], gp)
                _ψᶜ = get_ψᶜ(_λ, ξ̂ᶜ, Ψ)
            
                DT = id - ∇∇(_ψᶜ)
                J = abs(det(DT))
        
                for i in 1:nₘ, j in 1:nₘ, k in 1:nₘ
                    A[i,j,k] = - μ * sum(∫( ∇(ϕσ[i]) ⋅ inv(DT) ⋅ _w̄ * ϕσ[j] * ϕσ[k] * J )dΩ)
                end
            end
        
            return A
        end

        trbop = RBODEOperator{Float64,Nonlinear}(M, K, A, 1)

        ### initial condition
        _λ = get_λ( [t0, _ω], gp)
        _ψᶜ = get_ψᶜ(_λ, ξ̂ᶜ, Ψ)

        Tu₀(x) = nu₀( x - ∇(_ψᶜ)(x))

        ũ₀ = trbop.M(t0) \ [ sum(∫( Tu₀ * _ϕ * abs(det(id - ∇∇(_ψᶜ))) )dΩ) for _ϕ in ϕσ]
        
        # reconstruct FEM

        _uₕ = FEFunction(V₂, copy(ũ₀)' * get_free_dof_values.(ϕσ))
        _ψ = c̄_transform(_λ, ξᶜ, Ψ, V₁, N_fine, log_ūₕ_ref, MC)
        _uₕₜ_trb = pushfwd(_uₕ, _ψ, V₂, domain, dΩ)
    
        trb_ode_solver = ThetaMethod(nls, dt, θ)
        trbsol_t = solve(trb_ode_solver, trbop, ũ₀, t0, tF)

        push!(uₕₜ_trb, _uₕₜ_trb)
        push!(uₕₜ_trb_ref, _uₕ)

        for (_u,_t) in trbsol_t
            _λ = get_λ( [_t, _ω], gp)
            _ψᶜ = get_ψᶜ(_λ, ξ̂ᶜ, Ψ)

            _uₕ = FEFunction(V₂, copy(_u)' * get_free_dof_values.(ϕσ))
            _ψ = c̄_transform(_λ, ξᶜ, Ψ, V₁, N_fine, log_ūₕ_ref, MC)
            _uₕₜ_trb = pushfwd(_uₕ, _ψ, V₂, domain, dΩ)

            push!(uₕₜ_trb, _uₕₜ_trb)
            push!(uₕₜ_trb_ref, _uₕ)

        end
    end
    return uₕₜ_trb, uₕₜ_trb_ref
end