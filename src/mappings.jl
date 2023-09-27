"Push-forward operation for a function."
function pushfwd(u,ψ::FEFunction,V,dΩ)

    trian = get_triangulation(V)
    model = get_background_model(trian)
    
    a₁, b₁ = get_node_coordinates(get_grid(model))[1]
    a₂, b₂ = get_node_coordinates(get_grid(model))[end]

    p = Point((a₁ + a₂)/2,(b₁ + b₂)/2)
    
    u_cache = Gridap.CellData.return_cache(u, p)
    psi_cache = Gridap.CellData.return_cache(∇(ψ), p)

    function T★u(y)
        dy = Gridap.evaluate!(psi_cache, ∇(ψ), y)
        Ty = y - dy
        if Ty[1] < a₁ || Ty[1] > a₂ || Ty[2] < b₁ || Ty[2] > b₂
            return zero(typeof(Gridap.evaluate!(u_cache, u, y)))
        else
            return Gridap.evaluate!(u_cache, u, Ty)
        end
    end
    U = u.fe_space
    a(u,v) = ∫( u*v )dΩ
    l(v) = ∫( v*T★u )dΩ
    op = AffineFEOperator(a,l,U,V)
    return solve(op)
end

function pushfwd(u,ψ::Interpolations.Extrapolation,V,dΩ)

    trian = get_triangulation(V)
    model = get_background_model(trian)
    
    a₁, b₁ = get_node_coordinates(get_grid(model))[1]
    a₂, b₂ = get_node_coordinates(get_grid(model))[end]

    p = Point((a₁ + a₂)/2,(b₁ + b₂)/2)

    _ψ(x) = ψ(x...)
    T(y) = y - ∇(_ψ)(y)

    u_cache = Gridap.CellData.return_cache(u, p)
    
    function T★u(y)
        Ty = T(y)
        if Ty[1] < a₁ || Ty[1] > a₂ || Ty[2] < b₁ || Ty[2] > b₂
            return zero(typeof(Gridap.evaluate!(u_cache, u, p)))
        else
            return Gridap.evaluate!(u_cache, u, Ty)
        end
        #return Gridap.evaluate!(u_cache, u, T(y))
    end
    
    U = u.fe_space
    a(u,v) = ∫( u * v )dΩ
    l(v) = ∫( v * T★u )dΩ
    op = AffineFEOperator(a,l,U,V)
    return solve(op)
end

function get_transport_potential(μ, ξᶜ, Ψ, gp)
    ξ̂ᶜ = get_free_dof_values.(ξᶜ)
    λ = get_λ(μ, gp)
    ψᶜ = FEFunction(Ψ, λ' * ξ̂ᶜ)
    return ψᶜ
end

function get_transport_potential_∂t(t, ω, dt, ξᶜ, Ψ, gp)
    ξ̂ᶜ = get_free_dof_values.(ξᶜ)
    ∂tλ = ( get_λ( [t + dt/2, ω], gp) - get_λ( [t - dt/2, ω], gp) ) / dt
    ∂tψᶜ = FEFunction(Ψ, ∂tλ' * ξ̂ᶜ)
    return ∂tψᶜ
end

function boundary_projection(ψ̂ᶜ, δ⁻¹, κ, Ψ, dΩ, degree = 6)

    N_fine = size(ψ̂ᶜ,1) - 1
    trian = get_triangulation(get_cell_quadrature(dΩ))
    model = get_background_model(trian)

    Γ = BoundaryTriangulation(model)
    nb = get_normal_vector(Γ)
    dΓ = Measure(Γ,degree)

    a₁, b₁ = get_node_coordinates(get_grid(model))[1]
    a₂, b₂ = get_node_coordinates(get_grid(model))[end]

    xs = a₁:1/N_fine:a₂
    ys = b₁:1/N_fine:b₂

    ψᶜ_cubic_interpolant = cubic_spline_interpolation((xs, ys), ψ̂ᶜ)
    ψᶜ_eval = x -> ψᶜ_cubic_interpolant(x...)
    ψᶜ = interpolate_everywhere(ψᶜ_eval, Ψ)

    a(u,v) = ∫( (∇(v)⋅∇(u)) )dΩ + κ^2 * ∫( v*u )dΩ + δ⁻¹ * ∫( (nb⋅∇(u))*v )dΓ
    b(v) = ∫( (∇(v)⋅∇(ψᶜ)) )dΩ + κ^2 * ∫( v*ψᶜ )dΩ
    op = AffineFEOperator(a,b,TrialFESpace(Ψ),Ψ)

    return solve(op)
end