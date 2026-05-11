module Arnoldi

include("Orthogonalization.jl")
using .Orthogonalization

using LinearAlgebra

export arnoldi

function arnoldi(A; Q = nothing, H = nothing, start_at = 1, max_iter = 100, num_eig = 4, tol = 1e-10, arnoldi_relation=true)
    #= 
    The Arnoldi method for finding an orthonormal basis of the Krylov subspace and the upper Hessenberg matrix.
    Input: 
    - A: a matrix.
    - n: the size of the matrix.
    - Q: a matrix whose columns are orthonormal.
    - H: an upper Hessenberg matrix.
    - start_at: the index to start the Arnoldi iteration.
    - max_iter: the dimension of the Krylov subspace.
    - num_eig: the number of eigenvalues to compute.
    - tol: the tolerance for convergence.
    Output: the matrix Q, the upper Hessenberg matrix H, and the number of iterations.
    =#
    n = size(A, 1)

    if Q === nothing
        Q = zeros(ComplexF64, n, max_iter+1)
        Q[:, 1] = randn(n)
        Q[:, 1] /= norm(Q[:, 1])
    end
    if H === nothing
        H = zeros(ComplexF64, max_iter+1, max_iter)
    end

    for j in start_at:max_iter
        r = A * Q[:, j:j]
        r, h = Orthogonalization.orthogonalization(r, Q[:, 1:j], max_iter)
        H[1:j, j] = h[1:j]
        H[j+1, j] = norm(r)

        if norm(r) == 0
            return Q[:, 1:j+1], H[1:j+1, 1:j]
        end

        Q[:, 1+j] = r/norm(r)
        
        if j%20 == 0 && j > num_eig && arnoldi_relation
            evals, evecs = eigen(H[1:j, 1:j], sortby = x -> -abs(x))
            error = maximum(abs.(norm(r) .* evecs[end, 1:num_eig]))
            if error < tol
                return Q[:, 1:j+1], H[1:j+1, 1:j]
            end
        end
    end

    return Q, H
end

end