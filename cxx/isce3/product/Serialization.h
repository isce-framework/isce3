//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2019

/** \file product/Serialization.h
 *
 * Serialization functions for isce3::product objects. */

#pragma once

// isce3::core
#include <isce3/core/Constants.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/Serialization.h>

// isce3::io
#include <isce3/io/IH5.h>
#include <isce3/io/Serialization.h>

// isce3::product
#include <isce3/product/Metadata.h>
#include <isce3/product/Swath.h>

//! The isce namespace
namespace isce3 {
    //! The isce3::product namespace
    namespace product {

        /** Load ProcessingInformation from HDF5.
          *
          * @param[in] group        HDF5 group object.
          * @param[in] proc         ProcessingInformation object to be configured. */
        inline void loadFromH5(isce3::io::IGroup & group, ProcessingInformation & proc) {

            // Load slant range
            std::valarray<double> values;
            isce3::io::loadFromH5(group, "slantRange", values);
            proc.slantRange(values);

            // Load zero Doppler time
            isce3::io::loadFromH5(group, "zeroDopplerTime", values);
            proc.zeroDopplerTime(values);

            // Load effective velocity LUT
            isce3::core::LUT2d<double> lut;
            if (isce3::io::exists(group, "effectiveVelocity")) {
                isce3::core::loadCalGrid(group, "effectiveVelocity", lut);
                proc.effectiveVelocity(lut); 
            }

            if (isce3::io::exists(group, "frequencyA/azimuthFMRate")) {
                // Load azimuth FM rate and Doppler centroid for primary frequency (A)
                isce3::core::loadCalGrid(group, "frequencyA/azimuthFMRate", lut);
                proc.azimuthFMRate(lut, 'A');
            }
            isce3::core::loadCalGrid(group, "frequencyA/dopplerCentroid", lut);
            proc.dopplerCentroid(lut, 'A');

            // Check for existence of secondary frequencies
            if (isce3::io::exists(group, "frequencyB")) {

                if (isce3::io::exists(group, "frequencyB/azimuthFMRate")) {
                    isce3::core::loadCalGrid(group, "frequencyB/azimuthFMRate", lut);
                    proc.azimuthFMRate(lut, 'B');
                }
                isce3::core::loadCalGrid(group, "frequencyB/dopplerCentroid", lut);
                proc.dopplerCentroid(lut, 'B');
            }
        }

        /** Load Swath from HDF5
          *
          * @param[in] group        HDF5 group object.
          * @param[in] swath        Swath object to be configured. 
          * @param[in] freq         Frequency designation (e.g., A or B) */
        inline void loadFromH5(isce3::io::IGroup & group, Swath & swath, char freq) {

            // Open appropriate frequency group
            std::string freqString("frequency");
            freqString.push_back(freq);
            isce3::io::IGroup fgroup = group.openGroup(freqString);

            // Load slant range
            std::valarray<double> s_array;
            isce3::io::loadFromH5(fgroup, "slantRange", s_array);
            swath.slantRange(s_array);

            // Load zero Doppler time
            std::valarray<double> t_array;
            isce3::io::loadFromH5(group, "zeroDopplerTime", t_array);
            swath.zeroDopplerTime(t_array);

            // Get reference epoch
            isce3::core::DateTime refEpoch = isce3::io::getRefEpoch(group, "zeroDopplerTime");
            swath.refEpoch(refEpoch);

            // Load other parameters
            double value;
            isce3::io::loadFromH5(fgroup, "acquiredCenterFrequency", value);
            swath.acquiredCenterFrequency(value);

            isce3::io::loadFromH5(fgroup, "processedCenterFrequency", value);
            swath.processedCenterFrequency(value);

            isce3::io::loadFromH5(fgroup, "acquiredRangeBandwidth", value);
            swath.acquiredRangeBandwidth(value);

            isce3::io::loadFromH5(fgroup, "processedRangeBandwidth", value);
            swath.processedRangeBandwidth(value);

            isce3::io::loadFromH5(fgroup, "nominalAcquisitionPRF", value);
            swath.nominalAcquisitionPRF(value);

            isce3::io::loadFromH5(fgroup, "sceneCenterGroundRangeSpacing", value);
            swath.sceneCenterGroundRangeSpacing(value);

            isce3::io::loadFromH5(fgroup, "processedAzimuthBandwidth", value);
            swath.processedAzimuthBandwidth(value);
        }

        /** Load multiple swaths from HDF5
          *
          * @param[in] group            HDF5 group object.
          * @param[in] swaths           Map of Swaths to be configured. */
        inline void loadFromH5(isce3::io::IGroup & group, std::map<char, Swath> & swaths) {
            loadFromH5(group, swaths['A'], 'A');
            if (isce3::io::exists(group, "frequencyB")) {
                loadFromH5(group, swaths['B'], 'B');
            }
        }

        /** Load Metadata parameters from HDF5.
         *
         * @param[in] group         HDF5 group object.
         * @param[in] meta          Metadata object to be configured. */
        inline void loadFromH5(isce3::io::IGroup & group, Metadata & meta) {

            // Get orbit subgroup
            isce3::io::IGroup orbGroup = group.openGroup("orbit");
            // Configure a temporary orbit
            isce3::core::Orbit orbit;
            isce3::core::loadFromH5(orbGroup, orbit);
            // Save to metadata
            meta.orbit(orbit);            

            // Get attitude subgroup
            isce3::io::IGroup attGroup = group.openGroup("attitude");
            // Configure a temporary Euler angles object
            isce3::core::Attitude attitude;
            isce3::core::loadFromH5(attGroup, attitude);
            // Save to metadata
            meta.attitude(attitude);

            // Get processing information subgroup
            isce3::io::IGroup procGroup = group.openGroup("processingInformation/parameters");
            // Configure ProcessingInformation
            loadFromH5(procGroup, meta.procInfo());
        }

    }
}
