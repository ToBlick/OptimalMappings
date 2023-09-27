function get_λ(μ, gp)
    _μ = zeros((length(μ),1))
    _μ .= μ
    return [ predict_y(_gp, _μ)[1][1] for _gp in gp ]
end

"Gaussian process based on samples `_μ` and `λ`."
function get_gp(_μ, λ, m)
    mZero = MeanZero()
    kern = SE(0.0,0.0)
    logObsNoise = -6.0 

    gp = []
    for _m in 1:m
        _λ = [_λ[_m] for _λ in λ]

        _gp = GP(_μ, _λ, mZero, kern, logObsNoise)
        push!(gp, _gp)
    end
    return gp
end