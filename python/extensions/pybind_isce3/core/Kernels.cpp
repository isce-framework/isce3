#include "Kernels.h"
#include <pybind11/operators.h>
#include <isce/core/Kernels.h>

template class PyKernel<float>;
template class PyKernel<double>;

namespace py = pybind11;

using namespace isce::core;

// Base class, inherit from "trampoline" class to allow inheritance in python
template <typename T>
void addbinding(py::class_<Kernel<T>, PyKernel<T>> & pyKernel)
{
    pyKernel
        .def(py::init<>())
        .def("__call__", &Kernel<T>::operator())
        .def_property_readonly("width", &Kernel<T>::width);
}

template void addbinding(py::class_<Kernel<float>, PyKernel<float>> & pyKernel);
template void addbinding(py::class_<Kernel<double>, PyKernel<double>> & pyKernel);

// Bartlett
template <typename T>
void addbinding(py::class_<BartlettKernel<T>, Kernel<T>> & pyKernel)
{
    pyKernel.def(py::init<double>());
}

template void addbinding(py::class_<BartlettKernel<float>, Kernel<float>> & pyKernel);
template void addbinding(py::class_<BartlettKernel<double>, Kernel<double>> & pyKernel);

// Linear
template <typename T>
void addbinding(py::class_<LinearKernel<T>, Kernel<T>> & pyKernel)
{
    pyKernel.def(py::init<>());
}

template void addbinding(py::class_<LinearKernel<float>, Kernel<float>> & pyKernel);
template void addbinding(py::class_<LinearKernel<double>, Kernel<double>> & pyKernel);

// Knab
template <typename T>
void addbinding(py::class_<KnabKernel<T>, Kernel<T>> & pyKernel)
{
    pyKernel
        .def(py::init<double, double>())
        .def_property_readonly("bandwidth", &KnabKernel<T>::bandwidth);
}

template void addbinding(py::class_<KnabKernel<float>, Kernel<float>> & pyKernel);
template void addbinding(py::class_<KnabKernel<double>, Kernel<double>> & pyKernel);

// NFFT
template <typename T>
void addbinding(py::class_<NFFTKernel<T>, Kernel<T>> & pyKernel)
{
    pyKernel.def(py::init<int, int, int>());
}

template void addbinding(py::class_<NFFTKernel<float>, Kernel<float>> & pyKernel);
template void addbinding(py::class_<NFFTKernel<double>, Kernel<double>> & pyKernel);

// Look up table metakernel.  Allow double->float conversion.
void addbinding(py::class_<TabulatedKernel<float>, Kernel<float>> & pyKernel)
{
    pyKernel
        .def(py::init<const Kernel<float>&, int>())
        .def(py::init<const Kernel<double>&, int>());
}

void addbinding(py::class_<TabulatedKernel<double>, Kernel<double>> & pyKernel)
{
    pyKernel.def(py::init<const Kernel<double>&, int>());
}

// Chebyshev polynomial metakernel.  Allow double->float conversion.
void addbinding(py::class_<ChebyKernel<float>, Kernel<float>> & pyKernel)
{
    pyKernel
        .def(py::init<const Kernel<float>&, int>())
        .def(py::init<const Kernel<double>&, int>());
}

void addbinding(py::class_<ChebyKernel<double>, Kernel<double>> & pyKernel)
{
    pyKernel.def(py::init<const Kernel<double>&, int>());
}
