module Eigenpairs

include("Arnoldi.jl")
include("ImplicitRestart.jl")
include("BadRestart.jl")
using .Arnoldi
using .ImplicitRestart
using .BadRestart

using LinearAlgebra
using JacobiDavidson
using LinearMaps

export eigenpairs_arnoldi, eigenpairs_iram, eigenpairs_bad_restart, eigenpairs_jd

function eigenpairs_arnoldi(A, n; num_eig=n, subspace_dim=n, tol=1e-10)
    #= 
    The Arnoldi method for computing eigenvalues and eigenvectors of a matrix A.
    Input: 
    - A: a matrix.
    - n: the size of the matrix.
    - num_eig: the number of eigenvalues to compute.
    - max_iter: the dimension of the Krylov subspace.
    - tol: the tolerance for convergence.
    Output: the largest eigenvalues and eigenvectors of A.
    =#
    if num_eig > subspace_dim
        error("num_eig must be less than or equal to subspace_dim")
        return nothing
    end

    Q, H = Arnoldi.arnoldi(A; max_iter=subspace_dim, num_eig=num_eig, tol=tol)

    m = size(Q, 2) - 1
    Hm = H[1:m, 1:m]
    evals, evecs = eigen(Hm, sortby = x -> -abs(x))
    evecs = Q[:, 1:m] * evecs
    return evals[1:num_eig], evecs[:, 1:num_eig]
end

function eigenpairs_iram(A, n; num_eig=n, subspace_dim=n, restart_dim=n, max_iter=100, tol=1e-10)
    #= 
    The Implicit Restart Arnoldi method for computing eigenvalues and eigenvectors of a matrix A.
    Input: 
    - A: a matrix.
    - n: the size of the matrix.
    - num_eig: the number of eigenvalues to compute.
    - max_size: the maximum dimension of the Krylov subspace.
    - restart_dim: the dimension of the Krylov subspace after restart.
    - max_iter: the maximum number of restarts.
    - tol: the tolerance for convergence.
    Output: the largest eigenvalues and eigenvectors of A.
    =#
    if num_eig > subspace_dim || num_eig > restart_dim
        error("num_eig must be less than or equal to subspace_dim and restart_dim")
        return nothing
    end
    if restart_dim > subspace_dim
        error("restart_dim must be less than or equal to subspace_dim")
        return nothing
    end

    residuals = []

    Q, H = Arnoldi.arnoldi(A; max_iter=subspace_dim, num_eig=num_eig, tol=tol)

    for _ in 1:max_iter
        evals, evecs = eigen(H[1:end-1, :], sortby = x -> -abs(x))
        Q, H = ImplicitRestart.restart(Q, H, evals[restart_dim+1:end], restart_dim)

        Q, H = Arnoldi.arnoldi(A; Q=Q, H=H, start_at=restart_dim, max_iter=subspace_dim, num_eig=num_eig, tol=tol)

        m = size(Q, 2) - 1
        Hm = H[1:m, 1:m]
        evals, evecs = eigen(Hm, sortby = x -> -abs(x))
        error = maximum(abs.(H[m+1, m] .* evecs[end, 1:num_eig]))
        
        evecs = Q[:, 1:m] * evecs
        evals = evals[1:num_eig]
        evecs = evecs[:, 1:num_eig]
        residual = maximum(abs.(A * evecs - evecs * diagm(evals)))
        push!(residuals, residual)

        if error < tol
            break
        end
    end

    m = size(Q, 2) - 1
    Hm = H[1:m, 1:m]
    evals, evecs = eigen(Hm, sortby = x -> -abs(x))
    evecs = Q[:, 1:m] * evecs
    return evals[1:num_eig], evecs[:, 1:num_eig], residuals
end

function eigenpairs_naive_restart(A, n; num_eig=n, subspace_dim=n, restart_dim=n, max_iter=100, tol=1e-10)
    #= 
    The Implicit Restart Arnoldi method for computing eigenvalues and eigenvectors of a matrix A.
    Input: 
    - A: a matrix.
    - n: the size of the matrix.
    - num_eig: the number of eigenvalues to compute.
    - subspace_dim: the dimension of the subspace.
    - restart_dim: the dimension of the Krylov subspace after restart.
    - max_iter: the maximum number of restarts.
    - tol: the tolerance for convergence.
    Output: the largest eigenvalues and eigenvectors of A.
    =#
    if num_eig > subspace_dim || num_eig > restart_dim
        error("num_eig must be less than or equal to subspace_dim and restart_dim")
        return nothing
    end
    if restart_dim > subspace_dim
        error("restart_dim must be less than or equal to subspace_dim")
        return nothing
    end

    residuals = []

    Q, H = Arnoldi.arnoldi(A; max_iter=subspace_dim, num_eig=num_eig, tol=tol)

    m = size(Q, 2) - 1
    if m < subspace_dim
        Hm = H[1:m, 1:m]
        evals, evecs = eigen(Hm, sortby = x -> -abs(x))
        evecs = Q[:, 1:m] * evecs
        return evals[1:num_eig], evecs[:, 1:num_eig], residuals
    end

    for j in 1:max_iter
        Q, H = BadRestart.bad_restart(Q, H, restart_dim)
        
        Q, H = Arnoldi.arnoldi(A; Q=Q, H=H, start_at=restart_dim, max_iter=subspace_dim, num_eig=num_eig, tol=tol, arnoldi_relation=false)

        m = size(Q, 2) - 1
        Hm = H[1:m, 1:m]

        evals, evecs = eigen(Hm, sortby = x -> -abs(x))
        evecs = Q[:, 1:m] * evecs
        evals = evals[1:num_eig]
        evecs = evecs[:, 1:num_eig]
        error = maximum(abs.(A * evecs - evecs * diagm(evals)))
        push!(residuals, error)
        if error < tol
            break
        end
    end

    m = size(Q, 2) - 1
    Hm = H[1:m, 1:m]
    evals, evecs = eigen(Hm, sortby = x -> -abs(x))
    evecs = Q[:, 1:m] * evecs
    return evals[1:num_eig], evecs[:, 1:num_eig], residuals
end

function eigenpairs_jd(A, n; num_eig=n, max_iter=n, subspace_dim=n, restart_dim=n, tol=1e-10, verbosity=0)
    #= 
    The Jacobi-Davidson method for computing eigenvalues and eigenvectors of a matrix A.
    Input: 
    - A: a matrix.
    - n: the size of the matrix.
    - num_eig: the number of eigenvalues to compute.
    - max_iter: the maximum number of iterations.
    - subspace_dim: the dimension of the subspace.
    - restart_dim: the dimension of the Krylov subspace after restart.
    - tol: the tolerance for convergence.
    Output: the largest eigenvalues and eigenvectors of A.
    =#
    if num_eig > n
        error("num_eig must be less than or equal to n")
        return nothing
    end

    F = factorize(A)
    inverse_map = LinearMap(ComplexF64, size(A,1)) do y, x
        ldiv!(y, F, x)
    end

    schur, harmonic_ritz_values, converged_ritz_values, residuals = jdqr(inverse_map, target=Near(0.0 + 0im), subspace_dimensions=restart_dim:subspace_dim, max_iter=max_iter, pairs=num_eig, tolerance=tol, verbosity=verbosity)

    evals = schur.values
    evecs = schur.Q

    return evals, evecs, residuals
end

end