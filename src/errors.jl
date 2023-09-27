function rel_error_vec(u, v, err, dΩ)
    return [ err(u[i] - v[i], dΩ) / err(v[i], dΩ) for i in eachindex(u) ]
end

function abs_error_vec(u, v, err, dΩ)
    return [ err(u[i] - v[i], dΩ) for i in eachindex(u) ]
end

L2(e, dΩ) = sqrt(sum( ∫( e*e )dΩ ))
H1(e, dΩ) = sqrt(sum( ∫( e*e + ∇(e)⋅∇(e) )dΩ ))
Ḣ1(e, dΩ) = sqrt(sum( ∫( ∇(e)⋅∇(e) )dΩ ))
Ḣ1(u, ρ, dΩ) = sqrt(sum( ∫( ∇(u)⋅∇(u)*ρ )dΩ ))
Ḣ2(u) = sqrt(sum( ∫(∇∇(u) ⊙ ∇∇(u))dΩ ))