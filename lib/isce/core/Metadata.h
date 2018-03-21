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
    double radarWavelength, prf, rangeFirstSample, slantRangePixelSpacing, pegHeading;
    int lookSide;
    DateTime sensingStart;
    // Image formation related parameters
    int numberRangeLooks, numberAzimuthLooks;
    // Geometry parameters
    int width, length;
    // Basic constructor
    Metadata() : sensingStart() {}
};

// Define std::cout interaction for debugging
std::ostream& operator<<(std::ostream &os, const isce::core::Metadata &radar);

#endif

// end of file
