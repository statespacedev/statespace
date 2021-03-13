#ifndef PYBINDEXAMPLE_API_H
#define PYBINDEXAMPLE_API_H
#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

class Api {
public:
    Api();
    Eigen::MatrixXd test2(Eigen::MatrixXd matin);
    int test();
};

PYBIND11_MODULE(libstatespace, m) {
    pybind11::class_<Api>(m, "Api")
    .def(pybind11::init<>())
    .def("test", &Api::test)
    .def("test2", &Api::test2, pybind11::arg("matin"));
}

#endif //PYBINDEXAMPLE_API_H
