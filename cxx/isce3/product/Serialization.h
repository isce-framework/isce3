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
#include <isce3/product/Grid.h>

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

            isce3::io::loadFromH5(group, "zeroDopplerTimeSpacing", value);
            swath.zeroDopplerTimeSpacing(value);

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
            if (isce3::io::exists(group, "frequencyA")) {
                loadFromH5(group, swaths['A'], 'A');
            }
            if (isce3::io::exists(group, "frequencyB")) {
                loadFromH5(group, swaths['B'], 'B');
            }
        }

        /** Load Grid from HDF5
          *
          * @param[in] group        HDF5 group object.
          * @param[in] grid         Grid object to be configured. 
          * @param[in] freq         Frequency designation (e.g., A or B) */
        inline void loadFromH5(isce3::io::IGroup & group, Grid & grid, char freq) {

            // Open appropriate frequency group
            std::string freqString("frequency");
            freqString.push_back(freq);
            isce3::io::IGroup fgroup = group.openGroup(freqString);

            // Load X-coordinates
            std::valarray<double> x_array;
            isce3::io::loadFromH5(fgroup, "xCoordinates", x_array);
            grid.startX(x_array[0]);
            grid.width(x_array.size());

            // Load Y-coordinates
            std::valarray<double> y_array;
            isce3::io::loadFromH5(fgroup, "yCoordinates", y_array);
            grid.startY(y_array[0]);
            grid.length(y_array.size());

            // Get X-coordinate spacing
            double value;
            isce3::io::loadFromH5(fgroup, "xCoordinateSpacing", value);
            grid.spacingX(value);

            isce3::io::loadFromH5(fgroup, "yCoordinateSpacing", value);
            grid.spacingY(value);

            isce3::io::loadFromH5(fgroup, "rangeBandwidth", value);
            grid.rangeBandwidth(value);

            isce3::io::loadFromH5(fgroup, "azimuthBandwidth", value);
            grid.azimuthBandwidth(value);

            isce3::io::loadFromH5(fgroup, "centerFrequency", value);
            grid.centerFrequency(value);

            isce3::io::loadFromH5(fgroup, "slantRangeSpacing", value);
            grid.slantRangeSpacing(value);

            auto zero_dop_freq_vect = fgroup.find("zeroDopplerTimeSpacing", 
                                            ".", "DATASET");

            /* 
            Look for zeroDopplerTimeSpacing in frequency group
            (GCOV and GSLC products)
            */
            if (zero_dop_freq_vect.size() > 0) {
                isce3::io::loadFromH5(fgroup, "zeroDopplerTimeSpacing", value);
                grid.zeroDopplerTimeSpacing(value);
            } else {

                /* 
                Look for zeroDopplerTimeSpacing within grid group
                (GUNW products)
                */
                auto zero_dop_vect = group.find("zeroDopplerTimeSpacing", 
                                                ".", "DATASET");
                if (zero_dop_vect.size() > 0) {
                    isce3::io::loadFromH5(group, "zeroDopplerTimeSpacing", value);
                    grid.zeroDopplerTimeSpacing(value);
                } else {
                    grid.zeroDopplerTimeSpacing(
                        std::numeric_limits<double>::quiet_NaN());
                }
            }

            auto epsg_freq_vect = fgroup.find("epsg", ".", "DATASET");

            int epsg = -1;
            // Look for epsg in frequency group
            if (epsg_freq_vect.size() > 0) {
                isce3::io::loadFromH5(fgroup, "epsg", epsg);
                grid.epsg(epsg);
            } else {

                // Look for epsg in dataset projection
                auto projection_vect = fgroup.find("projection", 
                                                   ".", "DATASET");
                if (projection_vect.size() > 0) {
                    for (auto projection_str: projection_vect) {
                        auto projection_obj = fgroup.openDataSet(projection_str);
                        if (projection_obj.attrExists("epsg_code")) {
                            auto attr = projection_obj.openAttribute("epsg_code");
                            attr.read(getH5Type<int>(), &epsg);
                            grid.epsg(epsg);
                        }
                    }
                } 
            }
                
            if (epsg == -1) {
                throw isce3::except::RuntimeError(ISCE_SRCINFO(),
                        "ERROR could not infer EPSG code from input HDF5 file");
            }

        }

        /** Load multiple grids from HDF5
          *
          * @param[in] group            HDF5 group object.
          * @param[in] grids           Map of Grids to be configured. */
        inline void loadFromH5(isce3::io::IGroup & group, std::map<char, Grid> & grids) {
            if (isce3::io::exists(group, "frequencyA")) {
                loadFromH5(group, grids['A'], 'A');
            }
            if (isce3::io::exists(group, "frequencyB")) {
                loadFromH5(group, grids['B'], 'B');
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
