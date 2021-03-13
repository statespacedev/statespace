#ifndef PYBINDEXAMPLE_API_H
#define PYBINDEXAMPLE_API_H
#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

class Api {
public:
    Api();
    std::vector<Eigen::MatrixXd> udfactorize(Eigen::MatrixXd matin);
};

PYBIND11_MODULE(libstatespace, m) {
    pybind11::class_<Api>(m, "Api")
    .def(pybind11::init<>())
    .def("test", &Api::test)
    .def("udfactorize", &Api::udfactorize, pybind11::arg("matin"));
}

#endif //PYBINDEXAMPLE_API_H
