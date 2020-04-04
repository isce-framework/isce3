#include "DateTime.h"

#include <datetime.h>
#include <memory>
#include <pybind11/operators.h>
#include <string>

#include <isce/core/Kernels.h>
#include "Kernels.h"

template class PyKernel<float>;
template class PyKernel<double>;

namespace py = pybind11;

using namespace isce::core;

template <typename T>
void addbinding(py::class_<Kernel<T>, PyKernel<T>> & pyKernel)
{
    pyKernel
        .def("__call__", &Kernel<T>::operator())
        .def_property_readonly("width", &Kernel<T>::width)
        ;
}

template void addbinding(py::class_<Kernel<float>, PyKernel<float>> & pyKernel);
template void addbinding(py::class_<Kernel<double>, PyKernel<double>> & pyKernel);

template <typename T>
void addbinding(py::class_<BartlettKernel<T>, Kernel<T>> & pyKernel)
{
    pyKernel.def(py::init<double>());
}

template void addbinding(py::class_<BartlettKernel<float>, Kernel<float>> & pyKernel);
template void addbinding(py::class_<BartlettKernel<double>, Kernel<double>> & pyKernel);

template <typename T>
void addbinding(py::class_<LinearKernel<T>, Kernel<T>> & pyKernel)
{
    pyKernel.def(py::init<>());
}

template void addbinding(py::class_<LinearKernel<float>, Kernel<float>> & pyKernel);
template void addbinding(py::class_<LinearKernel<double>, Kernel<double>> & pyKernel);

template <typename T>
void addbinding(py::class_<KnabKernel<T>, Kernel<T>> & pyKernel)
{
    pyKernel
        .def(py::init<double, double>())
        .def_property_readonly("bandwidth", &KnabKernel<T>::bandwidth);
}

template void addbinding(py::class_<KnabKernel<float>, Kernel<float>> & pyKernel);
template void addbinding(py::class_<KnabKernel<double>, Kernel<double>> & pyKernel);
