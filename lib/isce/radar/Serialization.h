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

// isce::io
#include <isce/io/IH5.h>
#include <isce/io/Serialization.h>

// isce::radar
#include <isce/radar/Radar.h>

//! The isce namespace
namespace isce {
    //! The radar namespace
    namespace radar {

        /** Load Radar parameters from HDF5.
         *
         * @param[in] file          HDF5 file object.
         * @param[in] radar         Radar object to be configured. */
        void load(isce::io::IH5File & file, Radar & radar) {

            // Configure a temporary Poly2d object with data polynomial
            isce::core::Poly2d poly;
            isce::core::load(file, poly, "data_dcpolynomial");
            // Save to radar
            radar.contentDoppler(poly);

            // Configure Poly2d object with skew polynomial
            isce::core::load(file, poly, "skew_dcpolynomial");
            // Save to radar
            radar.skewDoppler(poly);
        }

    }
}

#endif

// end of file
