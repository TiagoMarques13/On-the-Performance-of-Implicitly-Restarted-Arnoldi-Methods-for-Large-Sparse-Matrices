module ImplicitRestart

using LinearAlgebra

function elementary(n, k)
    e = zeros(n, 1)
    e[k, 1] = 1
    return e
end

export restart

function restart(Q, H, mu, p)
    #= 
    The implicit restart procedure for the Arnoldi method.
    Input: 
    - Q: a matrix whose columns are orthonormal.
    - H: an upper Hessenberg matrix.
    - mu: the shifts used in the restart.
    - p: the dimension of the restarted subspace.
    Output: the matrix Q and the upper Hessenberg matrix H.
    =#
    Qm = Q[:, 1:end-1]
    Hm = H[1:end-1, :]
    
    v = elementary(size(Qm, 2), size(Qm, 2))'

    for i in 1:length(mu)
        F = qr(Hm - mu[i] * I)
        Li = Matrix(F.Q)
        Ri = F.R
        Hm = Ri * Li + mu[i] * I
        Qm = Qm * Li
        v = v * Li
    end
    r = Qm[:, p+1:p+1] * Hm[p+1:p+1, p:p] + Q[:, end] * H[end, end] * v[:, p:p]

    Q_final = zeros(ComplexF64, size(Q))
    Q_final[:, 1:p] = Qm[:, 1:p]
    Q_final[:, p+1] = r/norm(r)
    
    H_final = zeros(ComplexF64, size(H))
    H_final[1:p, 1:p] = Hm[1:p, 1:p]
    H_final[p+1, p] = norm(r)

    return Q_final, H_final
end

end