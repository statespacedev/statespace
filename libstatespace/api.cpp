#include "api.h"
#include <Eigen/Cholesky>

/*
 * class Api:
 *    '''handles calls from python code.'''
 * */
Api::Api() {}

/*
 *    def udfactorize(self):
 *       '''eigen ud factorization.'''
 * */
std::vector<Eigen::MatrixXd> Api::udfactorize(Eigen::MatrixXd M) {
    using namespace Eigen; std::vector<Eigen::MatrixXd> res;
    auto chol = M.ldlt(); res.emplace_back(chol.matrixU()); res.emplace_back(chol.vectorD());
    return res;
}

/*
 *    def temporal(self):
 *       '''thornton temporal update for the ud factorized ekf.'''
 * */
std::vector<Eigen::MatrixXd> Api::temporal(Eigen::MatrixXd xin, Eigen::MatrixXd Uin, Eigen::MatrixXd Din,
                                           Eigen::MatrixXd Phi, Eigen::MatrixXd Gin, Eigen::MatrixXd Q) {
    using namespace Eigen; std::vector<Eigen::MatrixXd> res;

}

//def temporal(self, xin, Uin, Din, Phi, Gin, Q):
//x, U, D = Phi @ xin, Uin, Din
//        n, r, G, U = 3, 3, Gin, np.eye(3)
//PhiU = Phi @ Uin
//for i in reversed(range(3)):
//sigma = 0
//for j in range(n):
//sigma = sigma + PhiU[i, j] ** 2 * Din[j, j]
//if (j <= r - 1): sigma = sigma + G[i, j] ** 2 + Q[j, j]
//D[i, i] = sigma
//        ilim = i - 1
//if not ilim < 0:
//for j in range(ilim):
//sigma = 0
//for k in range(n): sigma = sigma + PhiU[i, k] * Din[k, k] * PhiU[j, k]
//for k in range(r): sigma = sigma + G[i, k] * Q[k, k] * G[j, k]
//U[j, i] = sigma / D[i, i]
//for k in range(n): PhiU[j, k] = PhiU[j, k] - U[j, i] * PhiU[i, k]
//for k in range(r): G[j, k] = G[j, k] - U[j, i] * G[i, k]
//return x, U, D

