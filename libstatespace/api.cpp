#include "api.h"
#include <algorithm>
#include <iostream>
#include <map>

/*
 * class Api:
 *    '''handles calls from python code.'''
 * */
Api::Api() {}

/*
 *    def udfactorize(self):
 *       '''cpp implementation.'''
 * */
std::vector<Eigen::MatrixXd> Api::udfactorize(Eigen::MatrixXd matin) {

    std::vector<Eigen::MatrixXd> res; res.emplace_back(matin); res.emplace_back(matin);
    return res;
}

//def udfactorize(self, M):
//assert np.allclose(M, M.T)
//n, M = M.shape[0], np.triu(M)
//U, d = np.eye(n), np.zeros(n)
//for j in reversed(range(2, n + 1)):
//d[j - 1] = M[j - 1, j - 1]
//if d[j - 1] > 0: alpha = 1.0 / d[j - 1]
//else: alpha = 0.0
//for k in range(1, j):
//beta = M[k - 1, j - 1]
//U[k - 1, j - 1] = alpha * beta
//M[0:k, k - 1] = M[0:k, k - 1] - beta * U[0:k, j - 1]
//d[0] = M[0, 0]
//return U, np.diag(d)

