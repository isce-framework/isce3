//
// Source Author: Bryan Riel
// Co-Author: Joshua Cohen
// Copyright 2017

#ifndef __ISCE_CORE_METADATA_H__
#define __ISCE_CORE_METADATA_H__

#include <iostream>
#include "DateTime.h"

namespace isce { namespace core {
    struct RadarMetadata {
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
}}

std::ostream& operator<<(std::ostream &os, const isce::core::RadarMetadata &radar) {
    /*
     * Define the << operator for a Radar object for debugging purposes.
     */
    os << "Radar parameters:" << std::endl;
    os << "  - width: " << radar.width << std::endl;
    os << "  - length: " << radar.length << std::endl;
    os << "  - numberRangeLooks: " << radar.numberRangeLooks << std::endl;
    os << "  - numberAzimuthLooks: " << radar.numberAzimuthLooks << std::endl;
    os << "  - slantRangePixelSpacing: " << radar.slantRangePixelSpacing << std::endl;
    os << "  - rangeFirstSample: " << radar.rangeFirstSample << std::endl;
    os << "  - lookSide: " << radar.lookSide << std::endl;
    os << "  - prf: " << radar.prf << std::endl;
    os << "  - radarWavelength: " << radar.radarWavelength << std::endl;
    os << "  - sensingStart: " << radar.sensingStart.toIsoString() << std::endl;
    return os;
}

#endif
