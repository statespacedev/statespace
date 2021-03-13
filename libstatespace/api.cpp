#include "api.h"
#include <algorithm>
#include <iostream>
#include <map>

Api::Api() {}

int Api::test() {
    return 1; }

std::vector<Eigen::MatrixXd> Api::udfactorize(Eigen::MatrixXd matin) {
    std::vector<Eigen::MatrixXd> res;
    res.emplace_back(matin); res.emplace_back(matin);
    return res;
}

