module BadRestart

using LinearAlgebra

export bad_restart

function bad_restart(Q, H, r)
    #= 
    The naive restart procedure for the Arnoldi method.
    Input: 
    - Q: a matrix whose columns are orthonormal.
    - H: an upper Hessenberg matrix.
    - r: the dimension of the restarted subspace.
    Output: the matrix Q and the upper Hessenberg matrix H.
    =#
    Qm = Q[:, 1:end-1]
    Hm = H[1:end-1, :]

    lambda, V = eigen(Hm, sortby = x -> -abs(x))
    Lambda = diagm(lambda)

    V_tilde = V[:, 1:r]
    Lambda_tilde = Lambda[1:r, 1:r]

    Q_tilde, R_tilde = qr(V_tilde)
    Q_tilde = Q_tilde[:, 1:r]

    Q_r = Qm * Q_tilde
    H_r = R_tilde * Lambda_tilde * inv(R_tilde)

    Q_final = zeros(ComplexF64, size(Q))
    Q_final[:, 1:r] = Q_r
    Q_final[:, r+1] = Q[:, end]

    H_final = zeros(ComplexF64, size(H))
    H_final[1:r, 1:r] = H_r
    H_final[r+1, r] = H[end, end]

    return Q_final, H_final
end

end