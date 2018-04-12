// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Source Author: Bryan Riel
// Co-Author: Joshua Cohen
// Copyright 2017-2018

#ifndef ISCE_CORE_METADATA_H
#define ISCE_CORE_METADATA_H

#include <iosfwd>
#include "DateTime.h"

// Declarations
namespace isce {
    namespace core {
        struct Metadata;
    }
}

struct isce::core::Metadata {

    // Acquisition related parameters
    double radarWavelength;
    double prf;
    double rangeFirstSample;
    double slantRangePixelSpacing;
    double pulseDuration;
    double chirpSlope;
    double antennaLength;
    int lookSide;
    DateTime sensingStart;
    double pegHeading, pegLatitude, pegLongitude;

    // Image formation related parametesr
    int numberRangeLooks;
    int numberAzimuthLooks;

    // Geometry parameters
    int width;
    int length;

};

// Define std::cout interaction for debugging
std::ostream& operator<<(std::ostream &os, const isce::core::Metadata &radar);

#endif

// end of file
