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
    using namespace Eigen; std::vector<MatrixXd> res;
    MatrixXd U = MatrixXd::Identity(3, 3); auto D = Din; auto G = Gin; int n = 3; int r = 3;
    auto x = Phi * xin; MatrixXd PhiU = Phi * Uin;
    for (int i = 2; i > -1; --i) {
        int sigma = 0;
        for (int j = 0; j < n; ++ j) {
            sigma = sigma + pow(PhiU(i, j), 2) * Din(j, j);
            if (j <= r - 1) sigma = sigma + pow(G(i, j), 2) + Q(j, j); }
        D(i, i) = sigma;
        int ilim = i - 1;
        if (ilim > 0) {
            for (int j = 0; j < ilim; ++j) {
                int sigma = 0;
                for (int k = 0; k < n; ++k) sigma = sigma + PhiU(i, k) * Din(k, k) * PhiU(j, k);
                for (int k = 0; k < r; ++k) sigma = sigma + G(i, k) * Q(k, k) * G(j, k);
                U(j, i) = sigma / D(i, i);
                for (int k = 0; k < n; ++k) PhiU(j, k) = PhiU(j, k) - U(j, i) * PhiU(i, k);
                for (int k = 0; k < r; ++k) G(j, k) = G(j, k) - U(j, i) * G(i, k); } } }
    res.emplace_back(x); res.emplace_back(U); res.emplace_back(D);
    return res;
}

//      for k in range(n): PhiU[j, k] = PhiU[j, k] - U[j, i] * PhiU[i, k]
//      for k in range(r): G[j, k] = G[j, k] - U[j, i] * G[i, k]
//return x, U, D

