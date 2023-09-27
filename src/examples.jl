
abstract type AbstractProblem end

struct PoissonProblem{T} <: AbstractProblem 
    var::T
end

struct NonlinearAdvectionProblem{T, FT} <: AbstractProblem 
    β::T
    γ::T
    w̄::FT
end