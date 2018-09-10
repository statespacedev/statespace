import numpy as np

def UD(M):
    assert np.allclose(M, M.T)
    n = M.shape[0]
    M = np.triu(M)
    U = np.eye(n)
    d = np.zeros(n)
    for j in reversed(range(2, n + 1)):
        d[j - 1] = M[j - 1, j - 1]
        if d[j - 1] > 0:
            alpha = 1.0 / d[j - 1]
        else:
            alpha = 0.0
        for k in range(1, j):
            beta = M[k - 1, j - 1]
            U[k - 1, j - 1] = alpha * beta
            M[0:k, k - 1] = M[0:k, k - 1] - beta * U[0:k, j - 1]
    d[0] = M[0, 0]
    return U, np.diag(d)

def thornton(xin, Phi, Uin, Din, Gin, Q):
    x, U, D = Phi @ xin, Uin, Din
    n, r = 3, 3
    G = Gin
    U = np.eye(3)
    PhiU = Phi @ Uin
    for i in reversed(range(3)):
        sigma = 0
        for j in range(n):
            sigma = sigma + PhiU[i,j]**2 * Din[j,j]
            if (j <= r-1):
                sigma = sigma + G[i,j]**2 + Q[j,j]
        D[i,i] = sigma
        ilim = i-1
        if not ilim < 0:
            for j in range(ilim):
                sigma = 0
                for k in range(n):
                    sigma = sigma + PhiU[i,k] * Din[k,k] * PhiU[j,k]
                for k in range(r):
                    sigma = sigma + G[i,k] * Q[k,k] * G[j,k]
                U[j,i] = sigma / D[i,i]
                for k in range(n):
                    PhiU[j,k] = PhiU[j,k] - U[j,i] * PhiU[i,k]
                for k in range(r):
                    G[j,k] = G[j,k] - U[j,i] * G[i,k]
    return x, U, D

def bierman(z, R, H, xin, Uin, Din):
    x, U, D = xin, Uin, Din
    a = U.T @ H.T
    b = D @ a
    dz = z # z - H @ xin
    alpha = R
    gamma = 1 / alpha
    for j in range(3):
        beta = alpha
        alpha = alpha + a[j] * b[j]
        lamda = -a[j] * gamma
        gamma = 1 / alpha
        D[j, j] = beta * gamma * D[j, j]
        jlim = j-1
        if not jlim < 0:
            for i in range(jlim):
                beta = U[i, j]
                U[i, j] = beta + b[i] * lamda
                b[i] = b[i] + b[j] * beta
    dzs = gamma * dz
    x = x + dzs * b
    return x, U, D

def div0(a, b):
    try:
        return a / float(b)
    except:
        return np.nan