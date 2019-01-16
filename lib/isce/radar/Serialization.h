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

            // Load Doppler data
            std::valarray<double> r0, skew, content;
            isce::io::loadFromH5(dopGroup, "r0", r0);
            isce::io::loadFromH5(dopGroup, "skewdc_values", skew);
            isce::io::loadFromH5(dopGroup, "datadc_values", content);

            // Create a temporary Radar object
            Radar temp(r0, skew, content);

            // Copy to output Radar object
            radar = temp;
        }

    }
}

#endif

// end of file
