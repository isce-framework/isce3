// -*- C++ -*-
// -*- coding: utf-8 -*-
// 
// Author: Heresh Fattahi
// Copyright 2019-

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
    Phass() {
        _correlationThreshold = 0.2;
        _goodCorrelation = 0.7;
        _minPixelsPerRegion = 200.0;
        _usePower = true;
    };

    /** Destructor */
    ~Phass() = default;

    /** Unwrap the interferometric wrapped phase. */
    void unwrap(
        isce::io::Raster & phaseRaster,
        isce::io::Raster & corrRaster,
        isce::io::Raster & unwRaster,
        isce::io::Raster & labelRaster);

    /** Unwrap the interferometric wrapped phase. */
    void unwrap(
        isce::io::Raster & phaseRaster,
        isce::io::Raster & powerRaster,
        isce::io::Raster & corrRaster,
        isce::io::Raster & unwRaster,
        isce::io::Raster & labelRaster);

    /** Get correlation threshold increment. */
    double correlationThreshold() const;

    /** Set correlation threshold increment. */
    void correlationThreshold(const double);

    /** Get good correlation threshold. */
    double goodCorrelation() const;

    /** Set good correlation threshold. */
    void goodCorrelation(const double);

    /** Get minimum size of a region to be unwrapped. */
    int minPixelsPerRegion() const;

    /** Set minimum size of a region to be unwrapped. */
    void minPixelsPerRegion(const int);


    private:
        double _correlationThreshold = 0.2;
        double _goodCorrelation = 0.7; 
        int _minPixelsPerRegion = 200.0;
        bool _usePower = true;

};

}

// Get inline implementations.
#define ISCE_UNWRAP_PHASS_PHASS_ICC
#include "Phass.icc"
#undef ISCE_UNWRAP_PHASS_PHASS_ICC

#endif /* ISCE_UNWRAP_PHASS_PHASS_H */

