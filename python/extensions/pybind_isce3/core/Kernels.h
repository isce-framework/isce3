#pragma once

#include <isce3/core/Kernels.h>
#include <pybind11/pybind11.h>
#include <typeinfo>

// "Trampoline" class allowing inheritance in Python
template <typename T>
class PyKernel : public isce3::core::Kernel<T> {
public:
    using isce3::core::Kernel<T>::Kernel;
    PyKernel(double width) : isce3::core::Kernel<T>(width) {}

    T operator()(double x) const override {
        PYBIND11_OVERLOAD_PURE_NAME(T, isce3::core::Kernel<T>, "__call__", operator(), x);
    }
};

template <typename T>
inline bool is_cpp_kernel(const isce3::core::Kernel<T>& kernel)
{
    return typeid(kernel) != typeid(PyKernel<T>);
}

template <typename T>
void addbinding(pybind11::class_<isce3::core::Kernel<T>, PyKernel<T>> &);

template <typename T>
void addbinding(pybind11::class_<isce3::core::BartlettKernel<T>, isce3::core::Kernel<T>> &);

template <typename T>
void addbinding(pybind11::class_<isce3::core::LinearKernel<T>, isce3::core::Kernel<T>> &);

template <typename T>
void addbinding(pybind11::class_<isce3::core::KnabKernel<T>, isce3::core::Kernel<T>> &);

template <typename T>
void addbinding(pybind11::class_<isce3::core::AzimuthKernel<T>, isce3::core::Kernel<T>> &);

template <typename T>
void addbinding(pybind11::class_<isce3::core::NFFTKernel<T>, isce3::core::Kernel<T>> &);

void addbinding(pybind11::class_<isce3::core::TabulatedKernel<float>, isce3::core::Kernel<float>> &);
void addbinding(pybind11::class_<isce3::core::TabulatedKernel<double>, isce3::core::Kernel<double>> &);

void addbinding(pybind11::class_<isce3::core::ChebyKernel<float>, isce3::core::Kernel<float>> &);
void addbinding(pybind11::class_<isce3::core::ChebyKernel<double>, isce3::core::Kernel<double>> &);
