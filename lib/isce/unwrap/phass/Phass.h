#ifndef ISCE_UNWRAP_PHASS_PHASS_H
#define ISCE_UNWRAP_PHASS_PHASS_H

#include <array> // std::array
#include <complex> // std::complex
#include <cstddef> // size_t
#include <cstdint> // uint8_t

#include <isce/io/Raster.h> // isce::io::Raster

#include "PhassUnwrapper.h"

namespace isce::unwrap::phass
{

class Phass
{
public:
    /** Constructor */
    Phass() = default;

    /** Destructor */
    ~Phass() = default;

    void unwrap(
        isce::io::Raster & phaseRaster,
        isce::io::Raster & powerRaster,
        isce::io::Raster & corrRaster,
        isce::io::Raster & unwRaster,
        isce::io::Raster & labelRaster);

    private:
        double _corr_th = 0.2;
        double _good_corr = 0.7; 
        int _min_pixels_per_region = 20.0;

};

}

#endif /* ISCE_UNWRAP_PHASS_PHASS_H */

