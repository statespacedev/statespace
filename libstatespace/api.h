#ifndef PYBINDEXAMPLE_API_H
#define PYBINDEXAMPLE_API_H
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>

class Api {
public:
    Api();
    int test;
};

PYBIND11_MODULE(libstarid, m) {
    pybind11::class_<Api>(m, "Api")
    .def(pybind11::init<>())
    .def_readonly("test", &Api::test);
}

#endif //PYBINDEXAMPLE_API_H
