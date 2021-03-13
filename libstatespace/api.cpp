#include "api.h"
#include <Eigen/Cholesky>

/*
 * class Api:
 *    '''handles calls from python.'''
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
    return res; }

/*
 *    def observational(self):
 *       '''bierman observation update for the ud factorized ekf.'''
 * */
std::vector<Eigen::MatrixXd> Api::observational(Eigen::MatrixXd x, Eigen::MatrixXd U,
                                                Eigen::MatrixXd D, Eigen::MatrixXd H,
                                                double obs, double R, double yhat) {
    using namespace Eigen; std::vector<MatrixXd> res;
    auto dz = obs - yhat; auto alpha = R; auto gamma = 1/R;
    auto a = U.transpose() * H;
    Eigen::MatrixXd b = D * a;
    for (int j = 0; j < 3; ++j) {
        auto beta = alpha;
        alpha = alpha + a(j) * b(j);
        auto lambda = -a(j) * gamma;
        gamma = 1/alpha;
        D(j, j) = beta * gamma * D(j, j);
        int jlim = j - 1;
        if (jlim > 0) {
            for (int i = 0; i < jlim; ++i) {
                beta = U(i, j);
                U(i, j) = beta + b(i) * lambda;
                b(i) = b(i) + b(j) * beta; } } }
    x = x + gamma * dz * b;
    res.emplace_back(x); res.emplace_back(U); res.emplace_back(D);
    return res; }

