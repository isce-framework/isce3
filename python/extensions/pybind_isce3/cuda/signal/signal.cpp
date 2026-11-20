#include "signal.h"

#include "Crossmul.h"
#include "NFFT2d.h"

namespace py = pybind11;

void addsubmodule_cuda_signal(py::module & m)
{
    py::module m_signal = m.def_submodule("signal");

    // forward declare bound classes
    py::class_<isce3::cuda::signal::gpuCrossmul> pyCrossmul(m_signal, "Crossmul");
    py::class_<isce3::cuda::signal::NFFT2d<float>> pyNFFT2dF32(m_signal, "NFFT2dF32");
    py::class_<isce3::cuda::signal::NFFT2d<double>> pyNFFT2dF64(m_signal, "NFFT2dF64");
    py::class_<isce3::cuda::signal::NFFT2dResult<float>> pyNFFT2dF32Result(m_signal, "NFFT2dF32Result");
    py::class_<isce3::cuda::signal::NFFT2dResult<double>> pyNFFT2dF64Result(m_signal, "NFFT2dF64Result");

    // add bindings
    addbinding(pyCrossmul);
    addbinding<float>(pyNFFT2dF32);
    addbinding<double>(pyNFFT2dF64);
    addbinding<float>(pyNFFT2dF32Result);
    addbinding<double>(pyNFFT2dF64Result);
    addbinding_make_image_nfft2d_gpu(m_signal);
}
