//
// Source Author: Bryan Riel
// Co-Author: Joshua Cohen
// Copyright 2017

// double underbar compiler reserved namespace.  don't use it.
#if !defined(isce_core_metadata_h)
#define isce_core_metadata_h

#include <iosfwd>
#include "DateTime.h"

namespace isce {
    namespace core {
        struct RadarMetadata ;
    }
}

struct isce::core::RadarMetadata {
    // Acquisition related parameters
    double radarWavelength, prf, rangeFirstSample, slantRangePixelSpacing, pegHeading;
    int lookSide;
    DateTime sensingStart;
    // Image formation related parameters
    int numberRangeLooks, numberAzimuthLooks;
    // Geometry parameters
    int width, length;

    RadarMetadata() : sensingStart() {}
};

std::ostream& operator<<(std::ostream &os, const isce::core::RadarMetadata &radar);

#endif
