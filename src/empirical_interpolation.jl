struct EmpiricalInterpolation{DT, IT, FS}
    B::Matrix{DT}
    idxs::Vector{IT}
    A::Matrix{DT}
    W::FS
end

get_interpolation_matrix(E::EmpiricalInterpolation) = E.B
get_precomputed_matrix(E::EmpiricalInterpolation) = E.A
get_indices(E::EmpiricalInterpolation) = E.idxs
get_Q(E::EmpiricalInterpolation) = length(E.idxs)
fe_space(E::EmpiricalInterpolation) = E.W

function EmpiricalInterpolation(Ξ::Vector{<:FEFunction}, form, ϕ::Vector{<:FEFunction}, W::FESpace)
    Ξ̂ = get_cell_dof_values.(Ξ)
    Q = length(Ξ)
    B = zeros(Q,Q)
    X̂ = []
    i_magic = []
    for k in 1:Q
        r = Ξ̂[k]
        if k == 1
            nothing
        else
            θ = B[1:k-1,1:k-1] \ [ Ξ̂[k][i[1]][i[2]] for i in i_magic ]
            r -= θ' * X̂[1:k-1]
        end
        c̄ = argmax([ maximum(abs.(r[i])) for i in eachindex(r) ])
        ī = argmax(abs.(r[c̄]))
        i₁ = [c̄, ī]
        X₁ = r / r[c̄][ī]
        if k == 1
            X̂ = [X₁]
            i_magic = [i₁]
        else
            push!(X̂, X₁)
            push!(i_magic, i₁)
        end
        for i in 1:k
            for j in 1:k
                c̄ = i_magic[i][1]   # cell
                ī = i_magic[i][2]   # index
                B[i,j] = X̂[j][c̄][ī]
            end
        end
    end

    X = [ FEFunction(W, gather_free_values(W, X̂ᵢ)) for X̂ᵢ in X̂ ]

    A = form(ϕ, X)

    return EmpiricalInterpolation(B, i_magic, A, W)
end