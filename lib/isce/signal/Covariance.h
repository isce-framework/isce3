// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2018-
//

#ifndef ISCE_LIB_CROSSMUL_H
#define ISCE_LIB_CROSSMUL_H

# include <assert.h>

#include <isce/io/Raster.h>
#include <isce/core/LUT1d.h>
#include "Signal.h"
#include "Looks.h"

namespace isce {
    namespace signal {
        class Covariance;
    }
}

class isce::signal::Covariance {
    public:

        Covariance() {};

        ~Covariance() {};

        // Covariance estimation for dual-pol data (HH, VH)
        void covariance(isce::io::Raster& HH,
                isce::io::Raster& VH,
                isce::io::Raster& C11,
                isce::io::Raster& C12,
                isce::io::Raster& C22);

        void covariance(isce::io::Raster& HH,
                isce::io::Raster& VH,
                isce::io::Raster& HV,
                isce::io::Raster& VV,
                isce::io::Raster& C11,
                isce::io::Raster& C12,
                isce::io::Raster& C13,
                isce::io::Raster& C14,
                isce::io::Raster& C22,
                isce::io::Raster& C23,
                isce::io::Raster& C24,
                isce::io::Raster& C33,
                isce::io::Raster& C34,
                isce::io::Raster& C44);
    private:

        // number of range looks
        int _rangeLooks = 1;

        // number of azimuth looks
        int _azimuthLooks = 1;

        
}
