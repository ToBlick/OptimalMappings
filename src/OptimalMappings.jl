module OptimalMappings

    using LogExpFunctions
    using Distances
    using OptimalTransportTools
    using GaussianProcesses

    using Gridap, Gridap.FESpaces, Gridap.CellData
    using Gridap.CellData: get_cell_quadrature, get_node_coordinates

    using SparseArrays: spzeros

    using LinearAlgebra
    using LineSearches
    using ForwardDiff

    using Test
    using Statistics
    using Random

    using Interpolations

    include("examples.jl")
    export PoissonProblem, NonlinearAdvectionProblem

    include("empirical_interpolation.jl")
    export EmpiricalInterpolation, get_Q, get_interpolation_matrix, get_precomputed_matrix, get_indices 

    include("empirical_interpolation_forms.jl")
    export get_f★J, get_ā, get_ā★J, get_∂tΦ★J, get_J, get_K, form_f★J, form_K, form_ā, form_āu, form_J, form_∂tΦ★J, form_ā★J, form_āu★J

    include("errors.jl")
    export L2, H1, Ḣ1, Ḣ2, abs_error_vec, rel_error_vec

    include("gaussian_process.jl")
    export get_gp, get_λ

    include("rb_routines.jl")
    export pod, pod_monge_embedding

    include("ot_calculations.jl")
    export get_ρ̂, safe_log, get_ψ̂_ψ̂ᶜ, c_transform

    include("mappings.jl")
    export boundary_projection, pushfwd, get_transport_potential, get_transport_potential_∂t

    include("fe_spaces.jl")
    export initialize_fe_spaces

    include("example_routines.jl")
    export testparameterset, snapshot, snapshots

    include("rb_ode_operators.jl")
    export RBODEOperator, residual!, allocate_residual, jacobian!, jacobians!, allocate_jacobian, allocate_cache, update_cache!, get_order
end