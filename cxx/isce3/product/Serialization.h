//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2019

/** \file product/Serialization.h
 *
 * Serialization functions for isce3::product objects. */

#pragma once

#include <string>

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

#include <string>

//! The isce namespace
namespace isce3 {
    //! The isce3::product namespace
    namespace product {

        /** Load ProcessingInformation from HDF5.
          *
          * @param[in] group        HDF5 group object.
          * @param[in] proc         ProcessingInformation object to be configured. */
        inline void loadFromH5(isce3::io::IGroup & group, ProcessingInformation & proc) {

            // Load effective velocity LUT
            isce3::core::LUT2d<double> lut;
            if (isce3::io::exists(group, "effectiveVelocity")) {
                isce3::core::loadCalGrid(group, "effectiveVelocity", lut);
                proc.effectiveVelocity(lut); 
            }

            std::vector<std::string> frequencies{ "A", "B" };

            for (auto frequency : frequencies) {

                std::string frequency_str = "frequency" + frequency;
                // Check for existence of given frequency group
                if (isce3::io::exists(group, frequency_str)) {

                    // Get processing information subgroup

                    // First, try to read the coordinate vectors "zeroDopplerTime"
                    // and "slantRange" from the same group as the LUT.
                    // Check for the existence of the "zeroDopplerTime" H5 dataset in
                    // "frequency{frequency}"
                    if (isce3::io::exists(group, frequency_str+"/zeroDopplerTime")) {
                        isce3::io::IGroup freqGroup = group.openGroup(frequency_str);
                        if (isce3::io::exists(freqGroup, "azimuthFMRate")) {
                            // Load azimuth FM rate LUT (if available)
                            isce3::core::loadCalGrid(freqGroup, "azimuthFMRate", lut);
                            proc.azimuthFMRate(lut, frequency[0]);
                        }
                        if (isce3::io::exists(freqGroup, "dopplerCentroid")) {
                            // Load azimuth Doppler centroid LUT (if available)
                            isce3::core::loadCalGrid(freqGroup, "dopplerCentroid", lut);
                            proc.dopplerCentroid(lut, frequency[0]);
                        }
                    } 

                    // If not found, read the common coordinate vectors from the parent group.
                    // This supports legacy RSLC products (created prior to ~2020) and products
                    // created by nonofficial tools which may not support the current spec.
                    else {
                        if (isce3::io::exists(group, frequency_str+"/azimuthFMRate")) {
                            // Load azimuth FM rate LUT (if available)
                            isce3::core::loadCalGrid(group, frequency_str+"/azimuthFMRate", lut);
                            proc.azimuthFMRate(lut, frequency[0]);
                        }
                        if (isce3::io::exists(group, frequency_str+"/dopplerCentroid")) {
                            // Load azimuth Doppler centroid LUT (if available)
                            isce3::core::loadCalGrid(group, frequency_str+"/dopplerCentroid", lut);
                            proc.dopplerCentroid(lut, frequency[0]);
                        }
                    }
                }
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

            // Sub-swaths
            int num_of_sub_swaths = 1;
            if (isce3::io::exists(fgroup, "numberOfSubSwaths")) {
                isce3::io::loadFromH5(fgroup, "numberOfSubSwaths", num_of_sub_swaths);
            }

            swath.subSwaths().numSubSwaths(num_of_sub_swaths);

            for (int i=1; i<=num_of_sub_swaths; ++i) {
                if (!isce3::io::exists(fgroup, "validSamplesSubSwath" +
                                      std::to_string(i))) {
                    break;
                }

                std::vector<int> image_dims = getImageDims(fgroup,
                    "validSamplesSubSwath" + std::to_string(i));

                if (image_dims[0] != t_array.size()) {
                    std::string error_msg = "ERROR the valid-samples";
                    error_msg += " arrays for sub-swath " + std::to_string(i);
                    error_msg += " has length ";
                    error_msg += std::to_string(image_dims[0]);
                    error_msg += " whereas dataset zeroDopplerTime has";
                    error_msg += " length ";
                    error_msg += std::to_string(t_array.size());
                    throw isce3::except::RuntimeError(ISCE_SRCINFO(), error_msg);
                }

                isce3::core::Matrix<int> valid_samples_sub_swath_array(
                    image_dims[0], image_dims[1]);

                isce3::io::loadFromH5(
                    fgroup, "validSamplesSubSwath" + std::to_string(i),
                    valid_samples_sub_swath_array);

                swath.subSwaths().setValidSamplesArray(i, valid_samples_sub_swath_array);
            }

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

            isce3::io::loadFromH5(fgroup, "sceneCenterAlongTrackSpacing", value);
            swath.sceneCenterAlongTrackSpacing(value);

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

        /** Populate swath-related parameters of the Grid object from HDF5
          *
          * @param[in] group        HDF5 group object.
          * @param[in] grid         Grid object to be configured. 
          * @param[in] freq         Frequency designation (e.g., A or B) */
        inline void populateGridSwathParameterFromH5(isce3::io::IGroup & group, Grid & grid, const char freq) {

            // Open appropriate frequency group
            std::string freqString("frequency");
            freqString.push_back(freq);

            isce3::io::IGroup fgroup = group.openGroup(freqString);

            // Read other parameters
            double value;

            if (isce3::io::exists(fgroup, "rangeBandwidth")) {
                isce3::io::loadFromH5(fgroup, "rangeBandwidth", value);
                grid.rangeBandwidth(value);
            }

            if (isce3::io::exists(fgroup, "azimuthBandwidth")) {
                isce3::io::loadFromH5(fgroup, "azimuthBandwidth", value);
                grid.azimuthBandwidth(value);
            }

            if (isce3::io::exists(fgroup, "centerFrequency")) {
                isce3::io::loadFromH5(fgroup, "centerFrequency", value);
                grid.centerFrequency(value);
            }

            if (isce3::io::exists(fgroup, "slantRangeSpacing")) {
                isce3::io::loadFromH5(fgroup, "slantRangeSpacing", value);
                grid.slantRangeSpacing(value);
            }

            // Search for a dataset called "zeroDopplerTimeSpacing" in the "frequency{A,B}"
            // group.
            auto zero_dop_freq_vect = fgroup.find("zeroDopplerTimeSpacing",
                                                  ".", "DATASET");

            /* 
            GSLC products are expected to have 'zeroDopplerTimeSpacing' in the frequency group
            */
            if (zero_dop_freq_vect.size() > 0) {
                isce3::io::loadFromH5(fgroup, "zeroDopplerTimeSpacing", value);
                grid.zeroDopplerTimeSpacing(value);
            } else {

                /* 
                Look for zeroDopplerTimeSpacing within parent group
                (GCOV products)
                */
                auto zero_dop_vect = group.find("zeroDopplerTimeSpacing",
                                                ".", "DATASET");
                if (zero_dop_vect.size() > 0) {
                    isce3::io::loadFromH5(group, "zeroDopplerTimeSpacing", value);
                    grid.zeroDopplerTimeSpacing(value);
                }
                // If we don't find zeroDopplerTimeSpacing, we intentionally don't
                // set a value to not overwrite any existing value.
            }
        }

        /** Load Grid from HDF5
          *
          * @param[in] group        HDF5 group object.
          * @param[in] grid         Grid object to be configured. 
          * @param[in] freq         Frequency designation (e.g., A or B) */
        inline void loadFromH5(isce3::io::IGroup & group, Grid & grid, const char freq) {

            // Open appropriate frequency group
            std::string freqString("frequency");
            freqString.push_back(freq);
            isce3::io::IGroup fgroup = group.openGroup(freqString);

            // Get X and Y coordinates spacing
            double dx, dy;
            isce3::io::loadFromH5(fgroup, "xCoordinateSpacing", dx);
            grid.spacingX(dx);

            isce3::io::loadFromH5(fgroup, "yCoordinateSpacing", dy);
            grid.spacingY(dy);

            // Load X-coordinates
            std::valarray<double> x_array;
            isce3::io::loadFromH5(fgroup, "xCoordinates", x_array);
            grid.startX(x_array[0] - 0.5 * dx);
            grid.width(x_array.size());

            // Load Y-coordinates
            std::valarray<double> y_array;
            isce3::io::loadFromH5(fgroup, "yCoordinates", y_array);
            grid.startY(y_array[0] - 0.5 * dy);
            grid.length(y_array.size());

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

            populateGridSwathParameterFromH5(group, grid, freq);

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
        inline void loadFromH5(isce3::io::IGroup & group, Metadata & meta,
                               const std::string& product_level) {

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
            if (product_level == "L1") {
                isce3::io::IGroup procGroup = group.openGroup("processingInformation/parameters");
                // Configure ProcessingInformation
                loadFromH5(procGroup, meta.procInfo());
            }

            // Get processing information subgroup for GSLC and GCOV producs
            if (product_level == "L2" &&
                    isce3::io::exists(group, "sourceData/processingInformation/parameters")) {
                isce3::io::IGroup procGroup = group.openGroup("sourceData/processingInformation/parameters");
                // Configure ProcessingInformation
                loadFromH5(procGroup, meta.procInfo());
            }

        }

    }
}
