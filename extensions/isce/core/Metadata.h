//-*- C++ -*-
//-*- coding: utf-8 -*-

#ifndef ISCELIB_METADATA_H
#define ISCELIB_METADATA_H

#include <iostream>
#include <sstream>
#include "isce/core/DateTime.h"

namespace isce { namespace core {

    struct RadarMetadata {

        // Acquisition related parameters
        double radarWavelength;
        double prf;
        double rangeFirstSample;
        double slantRangePixelSpacing;
        int lookSide;
        isce::core::DateTime sensingStart;
        double pegHeading;

        // Image formation related parametesr
        int numberRangeLooks;
        int numberAzimuthLooks;

        // Geometry parameters
        int width;
        int length;

        // Constructors
        RadarMetadata() : sensingStart() {};

    }; // struct RadarMetadata

} // namespace core
} // namespace isce

// Declare the << operator for debugging purposes
extern std::ostream &operator<<(std::ostream & os,
    isce::core::RadarMetadata const & radar);


#endif

// end of file
