

#include "signalUtils.h"

#include <isce3/signal/Signal.h>

namespace isce3 { namespace signal {

template<class T_val>
void upsampleRasterBlockX(isce3::io::Raster& input_raster,
                          std::valarray<std::complex<T_val>>& output_array,
                          size_t offset_x, size_t offset_y, 
                          size_t input_size_x, size_t input_size_y,
                          size_t band, size_t upsample_factor)
{
    using T = std::complex<T_val>;
    size_t output_size_x = input_size_x * upsample_factor;
    size_t nthreads = 1;
    isce3::signal::Signal<T_val> refSignal(nthreads);
    std::valarray<T> slc(input_size_x * input_size_y);
    std::valarray<T> spectrum(input_size_x * input_size_y);
    std::valarray<T> spectrum_upsampled(output_size_x * input_size_y);
#pragma omp critical
    {
        input_raster.getBlock(&slc[0], offset_x, offset_y, input_size_x,
                              input_size_y, band);
    }
    // make forward and inverse fft plans for the reference SLC
    refSignal.forwardRangeFFT(slc, spectrum, input_size_x, input_size_y);
    refSignal.inverseRangeFFT(spectrum_upsampled, output_array, output_size_x,
                              input_size_y);
#pragma omp critical
    {
        refSignal.upsample(slc, output_array, input_size_y, input_size_x,
                           upsample_factor);
    }
}

template void
upsampleRasterBlockX<float>(isce3::io::Raster& input_raster,
                            std::valarray<std::complex<float>>& output_array,
                            size_t offset_x, size_t offset_y,
                            size_t input_size_x, size_t input_size_y,
                            size_t band, size_t upsample_factor);

template void upsampleRasterBlockX<double>(
        isce3::io::Raster& input_raster,
        std::valarray<std::complex<double>>& output_array, size_t offset_x,
        size_t offset_y, size_t input_size_x, size_t input_size_y, size_t band,
        size_t upsample_factor);

}}


