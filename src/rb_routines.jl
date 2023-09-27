"Obtain number of modes `n` from vector of eigenvalues `v` given tolerance `ϵ`."
function get_n(v, ϵ = 1e-3)
    for i in 1:length(v)
        if sum(v[1:i])/sum(v) > 1 - ϵ
            return i
        end
    end
end

"obtain `n` reduced bases from snapshots `uₕ` and `evd` of correlation matrix."
# obtain reduced basis from snapshots uₕ and evd of correlation matrix
function get_ϕ(n, uₕ, evd, V₂, U₂)
    fv = zero_free_values(U₂)
    _ϕ = FEFunction(V₂, fv)
    ϕ = Vector{typeof(_ϕ)}(undef, n)
    for i in 1:n
        fv .= 0
        for j in eachindex(uₕ)
            fv_j = get_free_dof_values(uₕ[j])
            fv .+= fv_j * evd.vectors[j,i]
        end
        fv ./= sqrt(evd.values[i])
        ϕ[i] = FEFunction(V₂, copy(fv))
    end
    return ϕ
end

function pod(u::Vector{<:FEFunction}, V::FESpace, U::FESpace, dΩ, ϵ)
    C = [ sum(∫(u[i] ⊙ u[j])dΩ) for i in eachindex(u), j in eachindex(u) ]
    evd = eigen(C, sortby = x -> -abs(x) )
    n = get_n(evd.values, ϵ)
    ϕ = get_ϕ(n, u, evd, V, U)
    return ϕ, evd
end

function pod_monge_embedding(u::Vector{<:FEFunction}, ρ_ref, V::FESpace, U::FESpace, dΩ, ϵ)
    C = [ sum(∫(∇(u[i])⋅∇(u[j])*ρ_ref)dΩ) for i in eachindex(u), j in eachindex(u) ]
    evd = eigen(C, sortby = x -> -abs(x) )
    n = get_n(evd.values, ϵ)
    ϕ = get_ϕ(n, u, evd, V, U)
    return ϕ, evd
end