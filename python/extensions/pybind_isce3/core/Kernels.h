#pragma once

#include <isce/core/Kernels.h>
#include <pybind11/pybind11.h>

// "Trampoline" class allowing inheritance in Python
template <typename T>
class PyKernel : public isce::core::Kernel<T> {
public:
    using isce::core::Kernel<T>::Kernel;
    PyKernel(double width) : isce::core::Kernel<T>(width) {}

    T operator()(double x) const override {
        PYBIND11_OVERLOAD_PURE_NAME(T, isce::core::Kernel<T>, "__call__", operator(), x);
    }
};

template <typename T>
void addbinding(pybind11::class_<isce::core::Kernel<T>, PyKernel<T>> &);

template <typename T>
void addbinding(pybind11::class_<isce::core::BartlettKernel<T>, isce::core::Kernel<T>> &);

template <typename T>
void addbinding(pybind11::class_<isce::core::LinearKernel<T>, isce::core::Kernel<T>> &);

template <typename T>
void addbinding(pybind11::class_<isce::core::KnabKernel<T>, isce::core::Kernel<T>> &);

template <typename T>
void addbinding(pybind11::class_<isce::core::NFFTKernel<T>, isce::core::Kernel<T>> &);

void addbinding(pybind11::class_<isce::core::TabulatedKernel<float>, isce::core::Kernel<float>> &);
void addbinding(pybind11::class_<isce::core::TabulatedKernel<double>, isce::core::Kernel<double>> &);

void addbinding(pybind11::class_<isce::core::ChebyKernel<float>, isce::core::Kernel<float>> &);
void addbinding(pybind11::class_<isce::core::ChebyKernel<double>, isce::core::Kernel<double>> &);
