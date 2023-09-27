function initialize_fe_spaces(N, N_fine, d, highorder, ::PoissonProblem, T = Float64)

    domain = Tuple(vcat([ [0,1] for _ in 1:d]...))
    partition = Tuple(N for _ in 1:d)
    model = CartesianDiscreteModel(domain,partition)
    V = FESpace(model, ReferenceFE(lagrangian,T,highorder), conformity=:H1, dirichlet_tags="boundary")
    U = TrialFESpace(V, x -> 0.0)
    Ω = Triangulation(model)
    degree = 2*highorder
    dΩ = Measure(Ω,degree)
    Γ = BoundaryTriangulation(model)
    nb = get_normal_vector(Γ)
    dΓ = Measure(Γ,degree)
    partition_fine = Tuple(N_fine for _ in 1:d)
    model_fine = CartesianDiscreteModel(domain,partition_fine)
    V_fine = FESpace(model_fine, ReferenceFE(lagrangian,T,1), conformity=:H1)
    Ψ = FESpace(model, ReferenceFE(lagrangian,T,3), conformity=:H1)
    W = Ψ
    W_matrix = TestFESpace(model, ReferenceFE(lagrangian,TensorValue{2,2,T,4},highorder), conformity=:H1)

    (
        domain = domain,
        partition = partition,
        model = model,
        V = V,
        U = U,
        Ω = Ω,
        degree = degree,
        dΩ = dΩ,
        Γ = Γ,
        nb = nb,
        dΓ = dΓ,
        partition_fine = partition_fine,
        model_fine = model_fine,
        V_fine = V_fine,
        Ψ = Ψ,
        W = W,
        W_matrix = W_matrix,
    )
end