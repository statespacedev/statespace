#ifndef PYBINDEXAMPLE_API_H
#define PYBINDEXAMPLE_API_H
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Core>

class Api {
public:
    Api()
};

PYBIND11_MODULE(cpp, m) {
    pybind11::class_<Api>(m, "Api")
            .def(pybind11::init<std::vector<std::string> &>());
}

#endif //PYBINDEXAMPLE_API_H
