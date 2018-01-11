//
// Source Author: Bryan Riel
// Co-Author: Joshua Cohen
// Copyright 2017
//

#include <iostream>
#include "Metadata.h"

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
    //os << "  - sensingStart: " << radar.sensingStart.toIsoString() << std::endl;
    return os;
}
