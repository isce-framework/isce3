#include <complex> // std::complex, std::conj, std::arg

#include "PhaseGrad.h" // calcPhaseGrad

namespace isce { namespace unwrap { namespace icu {

template<int WINSIZE>
void _calcPhaseGrad(
    float * phasegradx,
    float * phasegrady,
    const std::complex<float> * intf, 
    const size_t length,
    const size_t width)
{
    // Window weights (Gaussian kernel)
    float weights[WINSIZE * WINSIZE];
    float sum = 0.f;
    for (int jj = 0; jj < WINSIZE; ++jj)
    {
        for (int ii = 0; ii < WINSIZE; ++ii)
        {
            auto x = float(ii - WINSIZE/2);
            auto y = float(jj - WINSIZE/2);
            float w = exp(-(x*x + y*y) / (WINSIZE/2.f));
            weights[jj * WINSIZE + ii] = w;
            sum += w;
        }
    }
    for (int i = 0; i < WINSIZE * WINSIZE; ++i) { weights[i] /= sum; }

    // Init phase slope.
    const size_t tilesize = length * width;
    for (size_t i = 0; i < tilesize; ++i) 
    { 
        phasegradx[i] = 0.f; 
        phasegrady[i] = 0.f; 
    }

    // Compute smoothed phase slope using a weighted average of phase 
    // differences.
    #pragma omp parallel for collapse(2)
    for (size_t j = WINSIZE/2 + 1; j < length - WINSIZE/2; ++j)
    {
        for (size_t i = WINSIZE/2 + 1; i < width - WINSIZE/2; ++i)
        {
            auto sx = std::complex<float>(0.f, 0.f);
            auto sy = std::complex<float>(0.f, 0.f);

            #pragma unroll
            for (int jj = -WINSIZE/2; jj <= WINSIZE/2; ++jj)
            {
                #pragma unroll
                for (int ii = -WINSIZE/2; ii <= WINSIZE/2; ++ii)
                {
                    float w = weights[(jj + WINSIZE/2) * WINSIZE + (ii + WINSIZE/2)];

                    std::complex<float> z_11 = intf[(j+jj) * width + (i+ii)];
                    std::complex<float> z_10 = intf[(j+jj) * width + (i+ii-1)];
                    std::complex<float> z_01 = intf[(j+jj-1) * width + (i+ii)];

                    sx += w * z_11 * std::conj(z_10);
                    sy += w * z_11 * std::conj(z_01);
                }
            }

            phasegradx[j * width + i] = std::arg(sx);
            phasegrady[j * width + i] = std::arg(sy);
        }
    }
}

void _calcPhaseGrad(
    float * phasegradx,
    float * phasegrady,
    const std::complex<float> * intf, 
    const size_t length,
    const size_t width,
    const int winsize)
{
    // Window weights (Gaussian kernel)
    auto weights = new float[winsize * winsize];
    float sum = 0.f;
    for (int jj = 0; jj < winsize; ++jj)
    {
        for (int ii = 0; ii < winsize; ++ii)
        {
            auto x = float(ii - winsize/2);
            auto y = float(jj - winsize/2);
            float w = exp(-(x*x + y*y) / (winsize/2.f));
            weights[jj * winsize + ii] = w;
            sum += w;
        }
    }
    for (int i = 0; i < winsize * winsize; ++i) { weights[i] /= sum; }

    // Init phase slope.
    const size_t tilesize = length * width;
    for (size_t i = 0; i < tilesize; ++i) 
    { 
        phasegradx[i] = 0.f; 
        phasegrady[i] = 0.f; 
    }

    // Compute smoothed phase slope using a weighted average of phase 
    // differences.
    #pragma omp parallel for collapse(2)
    for (size_t j = winsize/2 + 1; j < length - winsize/2; ++j)
    {
        for (size_t i = winsize/2 + 1; i < width - winsize/2; ++i)
        {
            auto sx = std::complex<float>(0.f, 0.f);
            auto sy = std::complex<float>(0.f, 0.f);

            for (int jj = -winsize/2; jj <= winsize/2; ++jj)
            {
                for (int ii = -winsize/2; ii <= winsize/2; ++ii)
                {
                    float w = weights[(jj + winsize/2) * winsize + (ii + winsize/2)];

                    std::complex<float> z_11 = intf[(j+jj) * width + (i+ii)];
                    std::complex<float> z_10 = intf[(j+jj) * width + (i+ii-1)];
                    std::complex<float> z_01 = intf[(j+jj-1) * width + (i+ii)];

                    sx += w * z_11 * std::conj(z_10);
                    sy += w * z_11 * std::conj(z_01);
                }
            }

            phasegradx[j * width + i] = std::arg(sx);
            phasegrady[j * width + i] = std::arg(sy);
        }
    }

    delete[] weights;
}

void calcPhaseGrad(
    float * phasegradx,
    float * phasegrady,
    const std::complex<float> * intf, 
    const size_t length,
    const size_t width,
    const int winsize)
{
    // Let the compiler optimize for some common window sizes.
    switch(winsize)
    {
        case 3: 
            _calcPhaseGrad<3>(phasegradx, phasegrady, intf, length, width);
            break;
        case 5: 
            _calcPhaseGrad<5>(phasegradx, phasegrady, intf, length, width);
            break;
        case 7: 
            _calcPhaseGrad<7>(phasegradx, phasegrady, intf, length, width);
            break;
        case 9: 
            _calcPhaseGrad<9>(phasegradx, phasegrady, intf, length, width);
            break;
        case 11: 
            _calcPhaseGrad<11>(phasegradx, phasegrady, intf, length, width);
            break;
        default: 
            _calcPhaseGrad(phasegradx, phasegrady, intf, length, width, winsize);
            break;
    }
}

} } }

