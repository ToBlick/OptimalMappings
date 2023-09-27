
function form_f★J(ϕ, X, dΩ)
    return [sum(∫(Xᵢ * ϕⱼ)dΩ) for Xᵢ in X, ϕⱼ in ϕ]
end

function form_K(ϕ, X, dΩ)
    Q = length(X)
    nₘ = length(ϕ)
    A_tensor = zeros(Q, nₘ, nₘ)
    for k in 1:Q, i in 1:nₘ, j in 1:i
        A_tensor[k, i, j] = sum(∫(∇(ϕ[i]) ⋅ X[k] ⋅ ∇(ϕ[j]))dΩ)
        A_tensor[k, j, i] = A_tensor[k, i, j]
    end
    return reshape(A_tensor, Q, :)
end

function form_āu(ϕ, X, dΩ)
    Q = length(X)
    n = length(ϕ)
    A = zeros(Q, n, n, n)
    for q in 1:Q, i in 1:n, j in 1:n, k in 1:j
        A[q, i, j, k] = sum(∫(∇(ϕ[i]) ⋅ X[q] * ϕ[j] * ϕ[k])dΩ)
        A[q, i, k, j] = A[q, i, j, k]
    end
    return reshape(A, Q, :)
end

function form_ā(ϕ, X, dΩ)
    Q = length(X)
    n = length(ϕ)
    return reshape([sum(∫(∇(ϕ[i]) ⋅ X[q] * ϕ[j])dΩ) for q in 1:Q, i in 1:n, j in 1:n], Q, :)
end

function form_J(ϕ, X, dΩ)
    Q = length(X)
    n = length(ϕ)
    return reshape([sum(∫(ϕ[i] * ϕ[j] * X[q])dΩ) for q in 1:Q, i in 1:n, j in 1:n], Q, :)
end

function form_∂tΦ★J(ϕ, X, dΩ)
    Q = length(X)
    n = length(ϕ)
    return reshape([sum(∫(ϕ[i] * ∇(ϕ[j]) ⋅ X[q])dΩ) for q in 1:Q, i in 1:n, j in 1:n], Q, :)
end

function form_ā★J(ϕ, X, dΩ)
    Q = length(X)
    n = length(ϕ)
    return reshape([sum(∫(ϕ[j] * ∇(ϕ[i]) ⋅ X[q])dΩ) for q in 1:Q, i in 1:n, j in 1:n], Q, :)
end

function form_āu★J(ϕ, X, dΩ)
    Q = length(X)
    n = length(ϕ)
    A = zeros(Q, n, n, n)
    for q in 1:Q, i in 1:n, j in 1:n, k in 1:j
        A[q, i, j, k] = sum(∫(∇(ϕ[i]) ⋅ X[q] * ϕ[j] * ϕ[k])dΩ)
        A[q, i, k, j] = A[q, i, j, k]
    end
    return reshape(A, Q, :)
end


function get_f★J(f, μ, ψᶜ, W)
    x0 = μ .+ 0.5
    _f(x) = f(x, x0)
    psi_cache = Gridap.CellData.return_cache(∇(ψᶜ), Point(μ...))
    function f★(y)
        dy = Gridap.evaluate!(psi_cache, ∇(ψᶜ), y)
        Ty = y - dy
        return _f(Ty)
    end
    id = TensorValue(diagm(ones(2)))
    DT = id - ∇∇(ψᶜ)
    J = abs(det(DT))
    return interpolate_everywhere(f★ * J, W);
end

function get_K(ψᶜ, W)
    id = TensorValue(diagm(ones(2)))
    DT = id - ∇∇(ψᶜ)
    J = abs(det(DT))
    return interpolate_everywhere( inv(DT) ⋅ inv(DT) * J, W);
end

function get_J(ψᶜ, W)
    id = TensorValue(diagm(ones(2)))
    DT = id - ∇∇(ψᶜ)
    J = det(DT)
    return interpolate_everywhere(J, W)
end

function get_ā(w̄, α, W)
    _w̄(x) = w̄(x, α)
    return interpolate_everywhere(_w̄, W)
end

function get_ā★J(w̄, α, ψᶜ, W)
    id = TensorValue(diagm(ones(2)))
    DT = id - ∇∇(ψᶜ)
    J = abs(det(DT))
    _w̄(x) = w̄(x, α)
    return interpolate_everywhere( inv(DT) ⋅ _w̄ * J, W)
end

function get_∂tΦ★J(∂tψᶜ, ψᶜ, W)
    id = TensorValue(diagm(ones(2)))
    DT = id - ∇∇(ψᶜ)
    J = abs(det(DT))
    return interpolate_everywhere( inv(DT) ⋅ ∇(∂tψᶜ) * J, W)
end