#ifndef PYBINDEXAMPLE_API_H
#define PYBINDEXAMPLE_API_H
#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

class Api {
public:
    Api();
    std::vector<Eigen::MatrixXd> udfactorize(Eigen::MatrixXd M);
    std::vector<Eigen::MatrixXd> temporal(Eigen::MatrixXd xin, Eigen::MatrixXd Uin, Eigen::MatrixXd Din,
                                          Eigen::MatrixXd Phi, Eigen::MatrixXd Gin, Eigen::MatrixXd Q);
};

PYBIND11_MODULE(libstatespace, m) {
    pybind11::class_<Api>(m, "Api")
    .def(pybind11::init<>())
    .def("udfactorize", &Api::udfactorize, pybind11::arg("M"))
    .def("temporal", &Api::temporal, pybind11::arg("xin"), pybind11::arg("Uin"),
         pybind11::arg("Din"), pybind11::arg("Phi"),
         pybind11::arg("Gin"), pybind11::arg("Q"));
}

#endif //PYBINDEXAMPLE_API_H
