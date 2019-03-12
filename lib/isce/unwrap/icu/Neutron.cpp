#include <cmath> // sqrt
#include <complex> // std::abs
#include <exception> // std::out_of_range

#include "ICU.h" // ICU
#include "PhaseGrad.h" // calcPhaseGrad

namespace isce::unwrap::icu
{

void ICU::genNeutrons(
    bool * neut,
    const std::complex<float> * intf,
    const float * corr,
    const size_t length,
    const size_t width)
{
    // Init neutrons.
    const size_t tilesize = length * width;
    for (size_t i = 0; i < tilesize; ++i) { neut[i] = false; }

    if (_UsePhaseGradNeut)
    {
        // Compute phase gradient along range, azimuth.
        auto phasegradx = new float[tilesize];
        auto phasegrady = new float[tilesize];
        calcPhaseGrad(phasegradx, phasegrady, intf, length, width, _PhaseGradWinSize);

        // Get phase gradient neutrons.
        for (size_t i = 0; i < tilesize; ++i)
        { 
            neut[i] |= std::abs(phasegradx[i]) > _NeutPhaseGradThr;
        }

        delete[] phasegradx;
        delete[] phasegrady;
    }

    if (_UseIntensityNeut)
    {
        // Compute interferogram intensity.
        auto intensity = new float[tilesize];
        for (size_t i = 0; i < tilesize; ++i)
        {
            std::complex<float> z = intf[i];
            intensity[i] = z.real()*z.real() + z.imag()*z.imag();
        }
        
        // Estimate intensity mean and standard deviation using regularly 
        // sampled points. 
        constexpr size_t padx = 32;
        constexpr size_t pady = 16;
        constexpr size_t dx = 4;
        constexpr size_t dy = 4;

        if (width <= 2*padx || length <= 2*pady)
        {
            throw std::out_of_range("tile too small for intensity mean/stddev estimation");
        }

        float sum = 0.f;
        float sumSq = 0.f;
        size_t n = 0;
        for (size_t j = pady; j < length - pady; j += dy)
        {
            for (size_t i = padx; i < width - padx; i += dx)
            {
                float s = intensity[j * width + i];
                sum += s;
                sumSq += s*s;
                ++n;
            }
        }

        const float mu = sum / float(n);
        const float sigma = sqrt(sumSq / float(n) - (mu * mu));

        // Calculate adaptive intensity threshold based on tile mean and 
        // standard deviation.
        const float intensitythr = mu + _NeutIntensityThr * sigma;

        // Get intensity neutrons.
        for (size_t i = 0; i < tilesize; ++i)
        {
            neut[i] |= (intensity[i] > intensitythr) && (corr[i] < _NeutCorrThr);
        }

        delete[] intensity;
    }
}

}

