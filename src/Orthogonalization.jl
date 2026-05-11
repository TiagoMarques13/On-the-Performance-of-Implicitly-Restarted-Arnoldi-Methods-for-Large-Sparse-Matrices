module Orthogonalization

using LinearAlgebra

export orthogonalization

function orthogonalization(r, Q, m)
    #= 
    A function that orthogonalizes the vector r against the columns of Q and returns the orthogonalized vector and the coefficients h.
    Input: 
    - r: a vector.
    - Q: a matrix whose columns are orthogonal.
    Output: the new vector r and coefficients h.
    =#
    r = r
    h = zeros(ComplexF64, m)
    for i in 1:size(Q, 2)
        h[i] = dot(Q[:, i], r[:, 1])
        r .-= h[i] .* Q[:, i]
    end
    for i in 1:size(Q, 2)
        h[i] += dot(Q[:, i], r[:, 1])
        r .-= dot(Q[:, i], r[:, 1]) .* Q[:, i]
    end
    return r, h
end

end