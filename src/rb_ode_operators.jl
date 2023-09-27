import Gridap.ODEs.ODETools: ODEOperator
import Gridap.ODEs.ODETools: Nonlinear
import Gridap.ODEs.ODETools: allocate_cache
import Gridap.ODEs.ODETools: update_cache!
import Gridap.ODEs.ODETools: allocate_residual
import Gridap.ODEs.ODETools: jacobian!
import Gridap.ODEs.ODETools: jacobians!
import Gridap.ODEs.ODETools: allocate_jacobian
import Gridap.ODEs.ODETools: residual!

struct RBODEOperator{MT,KT,AT,C} <: ODEOperator{C}
    M::MT # mass matrix
    K::KT # linear part
    A::AT # quadratic part
    order::Int
end

RBODEOperator(M::MT, K::KT, A::AT) where {MT, KT, AT} = RBODEOperator{MT,KT,AT,Nonlinear}(M, K, A, 1)

get_order(op::RBODEOperator) = op.order
  
function residual!(r::AbstractVector,op::RBODEOperator,t::Real,x::NTuple{2,AbstractVector},ode_cache)
    u,u_t = x
    A_uu = zero(u)
    for j in eachindex(u), k in eachindex(u)
        A_uu .+= op.A(t)[:,j,k] * u[j] * u[k] 
    end
    r .= op.M(t) * u_t .+ op.K(t) * u .+ A_uu
end
  
function allocate_residual(op::RBODEOperator, u::AbstractVector, ode_cache)
    zero(u)
end

function allocate_residual(op::RBODEOperator, t0::Real, u::AbstractVector, ode_cache)
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
            for i in eachindex(u), j in eachindex(u), k in eachindex(u)
                A_u1[i,j] += op.A(t)[i,j,k] * u[k] # first index is test function, second is trial.
                A_u2[i,j] += op.A(t)[i,k,j] * u[k]
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
  
function allocate_jacobian(op::RBODEOperator, t0::Real, u::AbstractVector,cache)
    spzeros(size(op.M(0.0)))
end
  
allocate_cache(op::RBODEOperator) = nothing
allocate_cache(op::RBODEOperator,v::AbstractVector) = (similar(v),nothing)
allocate_cache(op::RBODEOperator,v::AbstractVector,a::AbstractVector) = (similar(v),similar(a),nothing)
update_cache!(cache,op::RBODEOperator,t::Real) = cache
  