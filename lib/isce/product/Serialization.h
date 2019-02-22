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

// isce::product
#include <isce/product/Metadata.h>
#include <isce/product/Swath.h>

//! The isce namespace
namespace isce {
    //! The isce::product namespace
    namespace product {

        /** Load ProcessingInformation from HDF5.
          *
          * @param[in] group        HDF5 group object.
          * @param[in] proc         ProcessingInformation object to be configured. */
        inline void loadFromH5(isce::io::IGroup & group, ProcessingInformation & proc) {

            // Load slant range
            std::valarray<double> values;
            isce::io::loadFromH5(group, "slantRange", values);
            proc.slantRange(values);

            // Load zero Doppler time
            isce::io::loadFromH5(group, "zeroDopplerTime", values);
            proc.zeroDopplerTime(values);

            // Load effective velocity LUT
            isce::core::LUT2d<double> lut;
            isce::core::loadCalGrid(group, "effectiveVelocity", lut);
            proc.effectiveVelocity(lut);

            // Load azimuth FM rate for each frequency
            isce::core::loadCalGrid(group, "frequencyA/azimuthFMRate", lut);
            proc.azimuthFMRate(lut, 'A');
            //isce::core::loadCalGrid(group, "frequencyB/azimuthFMRate", lut);
            //proc.azimuthFMRate(lut, 'B');

            // Load Doppler centroid for each frequency
            isce::core::loadCalGrid(group, "frequencyA/dopplerCentroid", lut);
            proc.dopplerCentroid(lut, 'A');
            //isce::core::loadCalGrid(group, "frequencyB/dopplerCentroid", lut);
            //proc.dopplerCentroid(lut, 'B');

        }

        /** Load Swath from HDF5
          *
          * @param[in] group        HDF5 group object.
          * @param[in] swath        Swath object to be configured. 
          * @param[in] freq         Frequency designation (e.g., A or B) */
        inline void loadFromH5(isce::io::IGroup & group, Swath & swath, char freq) {

            // Open appropriate frequency group
            std::string freqString("frequency");
            freqString.push_back(freq);
            isce::io::IGroup fgroup = group.openGroup(freqString);

            // Load slant range
            std::valarray<double> array;
            isce::io::loadFromH5(fgroup, "slantRange", array);
            swath.slantRange(array);

            // Load zero Doppler time
            isce::io::loadFromH5(group, "zeroDopplerTime", array);
            swath.zeroDopplerTime(array);

            // Get reference epoch
            isce::core::DateTime refEpoch = isce::io::getRefEpoch(group, "zeroDopplerTime");
            swath.refEpoch(refEpoch);

            // Load other parameters
            double value;
            isce::io::loadFromH5(fgroup, "acquiredCenterFrequency", value);
            swath.acquiredCenterFrequency(value);

            isce::io::loadFromH5(fgroup, "processedCenterFrequency", value);
            swath.processedCenterFrequency(value);

            isce::io::loadFromH5(fgroup, "acquiredRangeBandwidth", value);
            swath.acquiredRangeBandwidth(value);

            isce::io::loadFromH5(fgroup, "processedRangeBandwidth", value);
            swath.processedRangeBandwidth(value);

            isce::io::loadFromH5(fgroup, "nominalAcquisitionPRF", value);
            swath.nominalAcquisitionPRF(value);

            isce::io::loadFromH5(fgroup, "sceneCenterGroundRangeSpacing", value);
            swath.sceneCenterGroundRangeSpacing(value);

            isce::io::loadFromH5(fgroup, "processedAzimuthBandwidth", value);
            swath.processedAzimuthBandwidth(value);
        }

        /** Load multiple swaths from HDF5
          *
          * @param[in] group            HDF5 group object.
          * @param[in] swaths           Map of Swaths to be configured. */
        inline void loadFromH5(isce::io::IGroup & group, std::map<char, Swath> & swaths) {
            loadFromH5(group, swaths['A'], 'A');
            //loadFromH5(group, swaths['B'], 'B');
        }

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
            loadFromH5(procGroup, meta.procInfo());
        }

    }
}

#endif

// end of file
