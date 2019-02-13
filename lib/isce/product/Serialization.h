//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2019

/** \file Serialization.h
 *
 * Serialization functions for isce::product objects. */

#ifndef ISCE_PRODUCT_SERIALIZATION_H
#define ISCE_PRODUCT_SERIALIZATION_H

// isce::core
#include <isce/core/Constants.h>
#include <isce/core/Ellipsoid.h>
#include <isce/core/Serialization.h>

// isce::io
#include <isce/io/IH5.h>
#include <isce/io/Serialization.h>

// isce::radar
#include <isce/radar/Serialization.h>

// isce::product
#include <isce/product/ComplexImagery.h>
#include <isce/product/ImageMode.h>
#include <isce/product/Identification.h>
#include <isce/product/Metadata.h>


//! The isce namespace
namespace isce {
    //! The isce::product namespace
    namespace product {

        /** Load Metadata parameters from HDF5.
         *
         * @param[in] group         HDF5 group object.
         * @param[in] meta          Metadata object to be configured. */
        inline void loadFromH5(isce::io::IGroup & group, Metadata & meta) {

            // Get orbit subgroup
            isce::io::IGroup orbGroup = group.openGroup("orbit");
            // Configure a temporary orbit
            isce::core::Orbit orbit;
            isce::core::loadFromH5(orbGroup, orbit);
            // Save to metadata
            meta.orbit(orbit);            

            // Get attitude subgroup
            isce::io::IGroup attGroup = group.openGroup("attitude");
            // Configure a temporary Euler angles object
            isce::core::EulerAngles euler;
            isce::core::loadFromH5(attGroup, euler);
            // Save to metadata
            meta.attitude(euler);

            // Get processing information subgroup
            isce::io::IGroup procGroup = group.openGroup("processingInformation/parameters");
            // Configure ProcessingInformation
            isce::core::loadFromH5(procGroup, meta.procInfo());
        }

        /** Load ProcessingInformation from HDF5.
          *
          * @param[in] group        HDF5 group object.
          * @param[in] proc         ProcessingInformation object to be configured. */
        inline void loadFromH5((isce::io::IGroup & group, ProcessingInformation & proc) {

            // Load slant range
            std::valarray<double> values;
            isce::io::loadFromH5(group, "slantRange", values);
            proc.slantRange(values);

            // Load zero Doppler time
            isce::io::loadFromH5(group, "zeroDopplerTime", values);
            proc.zeroDopplerTime(values);

            // Load effective velocity
            isce::core::LUT2d<double> lut;
            isce::core::loadFromH5(group, "effectiveVelocity", lut);
            proc.effectiveVelocity(lut);

            // Load azimuth FM rate for each frequency
            isce::core::loadFromH5(group, "frequencyA/azimuthFMRate", lut);
            proc.azimuthFMRate(lut, 'A');
            isce::core::loadFromH5(group, "frequencyB/azimuthFMRate", lut);
            proc.azimuthFMRate(lut, 'B');

            // Load Doppler centroid for each frequency
            isce::core::loadFromH5(group, "frequencyA/dopplerCentroid", lut);
            proc.dopplerCentroid(lut, 'A');
            isce::core::loadFromH5(group, "frequencyB/dopplerCentroid", lut);
            proc.dopplerCentroid(lut, 'B');

        }
        
    }
}

#endif

// end of file
