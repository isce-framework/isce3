#include "Kernels.h"
#include <pybind11/operators.h>
#include <isce3/core/Kernels.h>

template class PyKernel<float>;
template class PyKernel<double>;

namespace py = pybind11;

using namespace isce::core;

// Base class, inherit from "trampoline" class to allow inheritance in python
template <typename T>
void addbinding(py::class_<Kernel<T>, PyKernel<T>> & pyKernel)
{
    pyKernel.doc() = "A function defined over the domain [-width/2, width/2]";
    pyKernel
        .def(py::init<double>(), py::arg("width"))
        .def("__call__", &Kernel<T>::operator())
        .def_property_readonly("width", &Kernel<T>::width);
}

template void addbinding(py::class_<Kernel<float>, PyKernel<float>> & pyKernel);
template void addbinding(py::class_<Kernel<double>, PyKernel<double>> & pyKernel);

// Bartlett
template <typename T>
void addbinding(py::class_<BartlettKernel<T>, Kernel<T>> & pyKernel)
{
    pyKernel.doc() = "Bartlett / Triangular kernel";
    pyKernel.def(py::init<double>());
}

template void addbinding(py::class_<BartlettKernel<float>, Kernel<float>> & pyKernel);
template void addbinding(py::class_<BartlettKernel<double>, Kernel<double>> & pyKernel);

// Linear
template <typename T>
void addbinding(py::class_<LinearKernel<T>, Kernel<T>> & pyKernel)
{
    pyKernel.doc() = "Linear interpolation kernel.";
    pyKernel.def(py::init<>());
}

template void addbinding(py::class_<LinearKernel<float>, Kernel<float>> & pyKernel);
template void addbinding(py::class_<LinearKernel<double>, Kernel<double>> & pyKernel);

// Knab
template <typename T>
void addbinding(py::class_<KnabKernel<T>, Kernel<T>> & pyKernel)
{
    pyKernel.doc() = "Knab (1983) kernel.  Bandwidth [0,1) in cycles/sample.";
    pyKernel
        .def(py::init<double, double>(), py::arg("width"), py::arg("bandwidth"))
        .def_property_readonly("bandwidth", &KnabKernel<T>::bandwidth);
}

template void addbinding(py::class_<KnabKernel<float>, Kernel<float>> & pyKernel);
template void addbinding(py::class_<KnabKernel<double>, Kernel<double>> & pyKernel);

// NFFT
template <typename T>
void addbinding(py::class_<NFFTKernel<T>, Kernel<T>> & pyKernel)
{
    pyKernel.doc() = R"delim(
        NFFT time-domain kernel.

        This is called phi(x) in the NFFT papers (Keiner 2009),
        specifically the Kaiser-Bessel window function.
        The domain is scaled so that usage is the same as other ISCE kernels, e.g.,
        for x in [0,n) instead of [-0.5,0.5).
    )delim";
    pyKernel.def(py::init<int, int, int>(), py::arg("halfwidth"),
        py::arg("size_data"), py::arg("size_fft"));
}

template void addbinding(py::class_<NFFTKernel<float>, Kernel<float>> & pyKernel);
template void addbinding(py::class_<NFFTKernel<double>, Kernel<double>> & pyKernel);

static const auto tabdoc = R"(
    Kernel look-up table.  Initialized from another kernel, which is assumed to
    have even symmetry.
)";

// Look up table metakernel.  Allow double->float conversion.
void addbinding(py::class_<TabulatedKernel<float>, Kernel<float>> & pyKernel)
{
    pyKernel.doc() = tabdoc;
    pyKernel
        .def(py::init<const Kernel<float>&, int>(),
            py::arg("kernel"), py::arg("table_size"))
        .def(py::init<const Kernel<double>&, int>(),
            py::arg("kernel"), py::arg("table_size"));
}

void addbinding(py::class_<TabulatedKernel<double>, Kernel<double>> & pyKernel)
{
    pyKernel.doc() = tabdoc;
    pyKernel.def(py::init<const Kernel<double>&, int>(),
        py::arg("kernel"), py::arg("table_size"));
}

static const auto chebdoc = R"(
    Kernel polynomial approximation.  Initialized from another kernel, which is
    assumed to have even symmetry.
)";

// Chebyshev polynomial metakernel.  Allow double->float conversion.
void addbinding(py::class_<ChebyKernel<float>, Kernel<float>> & pyKernel)
{
    pyKernel.doc() = chebdoc;
    pyKernel
        .def(py::init<const Kernel<float>&, int>(),
            py::arg("kernel"), py::arg("num_coeff"))
        .def(py::init<const Kernel<double>&, int>(),
            py::arg("kernel"), py::arg("num_coeff"));
}

void addbinding(py::class_<ChebyKernel<double>, Kernel<double>> & pyKernel)
{
    pyKernel.doc() = chebdoc;
    pyKernel.def(py::init<const Kernel<double>&, int>(),
        py::arg("kernel"), py::arg("num_coeff"));
}
