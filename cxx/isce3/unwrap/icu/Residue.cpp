#include <cmath> // round

#include "ICU.h" // ICU

namespace isce::unwrap::icu
{

void ICU::getResidues(
    signed char * charge, 
    const float * phase, 
    const size_t length, 
    const size_t width)
{
    constexpr float twopi = 2.f * M_PI;

    // Get residue charge at each pixel (except last row & col).
    for (size_t j = 0; j < length-1; ++j)
    {
        for (size_t i = 0; i < width-1; ++i)
        {
            // Compute path integral around a 4 pixel neighborhood.
            float phi_00 = phase[(j+0) * width + (i+0)];
            float phi_10 = phase[(j+1) * width + (i+0)];
            float phi_01 = phase[(j+0) * width + (i+1)];
            float phi_11 = phase[(j+1) * width + (i+1)];

            charge[j * width + i] = round((phi_10 - phi_00) / twopi) + 
                                    round((phi_11 - phi_10) / twopi) + 
                                    round((phi_01 - phi_11) / twopi) + 
                                    round((phi_00 - phi_01) / twopi);
        }
    }

    // Set last row & col's charge to zero.
    for (size_t i = 0; i < width; ++i) { charge[(length-1) * width + i] = 0; }
    for (size_t j = 0; j < length; ++j) { charge[j * width + (width-1)] = 0; }
}

}

