//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018

/** \file Serialization.h
 *
 * Serialization functions for isce::radar objects. */

#ifndef ISCE_RADAR_SERIALIZATION_H
#define ISCE_RADAR_SERIALIZATION_H

// isce::core
#include <isce/core/Serialization.h>

// isce::io
#include <isce/io/IH5.h>

// isce::radar
#include <isce/radar/Radar.h>

//! The isce namespace
namespace isce {
    //! The radar namespace
    namespace radar {

        /** Load Radar parameters from HDF5.
         *
         * @param[in] group         HDF5 group object.
         * @param[in] radar         Radar object to be configured. */
        inline void loadFromH5(isce::io::IGroup & group, Radar & radar) {

            // Get doppler subgroup
            isce::io::IGroup dopGroup = group.openGroup("doppler_centroid");

            // Configure a temporary Poly2d object with data polynomial
            isce::core::Poly2d cpoly;
            isce::core::loadFromH5(dopGroup, cpoly, "data_dcpolynomial");
            // Save to radar
            radar.contentDoppler(cpoly);

            // Configure Poly2d object with skew polynomial
            isce::core::Poly2d spoly;
            isce::core::loadFromH5(dopGroup, spoly, "skew_dcpolynomial");
            // Save to radar
            radar.skewDoppler(spoly);
        }

    }
}

#endif

// end of file
