//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018

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

        /** Load Identification parameters from HDF5.
         *
         * @param[in] group     HDF5 group object.
         * @param[in] id        Identification object to be configured. */
        inline void loadFromH5(isce::io::IGroup & group, Identification & id) {

            // Configure a temporary ellipsoid
            isce::core::Ellipsoid ellps;
            isce::core::loadFromH5(group, ellps);
            // Save to identification
            id.ellipsoid(ellps);

            // Load the look direction
            isce::core::FixedString lookDir;
            isce::io::loadFromH5(group, "look_direction", lookDir);
            // Save to identification
            id.lookDirection(std::string(lookDir.str));
        }

        /** Load Metadata parameters from HDF5.
         *
         * @param[in] group         HDF5 group object.
         * @param[in] meta          Metadata object to be configured. */
        inline void loadFromH5(isce::io::IGroup & group, Metadata & meta) {

            // Get orbit subgroup
            isce::io::IGroup orbGroup = group.openGroup("orbit");

            // Configure a temporary orbit with NOE data
            isce::core::Orbit orbit;
            isce::core::loadFromH5(orbGroup, orbit, "NOE");
            // Save to metadata
            meta.orbitNOE(orbit);

            // Configure a temporary orbit with POE data
            isce::core::loadFromH5(orbGroup, orbit, "POE");
            // Save to metadata
            meta.orbitPOE(orbit);

            // Open instrument subgroup
            isce::io::IGroup instGroup = group.openGroup("instrument_data");

            // Configure a temporary instrument
            isce::radar::Radar instrument;
            isce::radar::loadFromH5(instGroup, instrument);
            // Save to metadata
            meta.instrument(instrument);

            // Open identification subgroup
            isce::io::IGroup idGroup = group.openGroup("identification");

            // Configure a temporary identification
            Identification id;
            loadFromH5(idGroup, id);
            // Save to metadata
            meta.identification(id);
        }

        /** Load ImageMode data from HDF5.
         *
         * @param[in] group         HDF5 group object.
         * @param[in] mode          ImageMode object to be configured. 
         * @param[in] modeType      String representing mode type. */
        inline void loadFromH5(isce::io::IGroup & group, ImageMode & mode,
                         const std::string & modeType) {

            // Get subgroup for specific image mode
            isce::io::IGroup modeGroup = group.openGroup(modeType + "_mode");
            // Set mode type
            mode.modeType(modeType);

            // Set the image dimensions
            std::vector<int> dims = isce::io::getImageDims(modeGroup, "hh");
            std::array<size_t, 2> arrayDims{static_cast<size_t>(dims[0]),
                                            static_cast<size_t>(dims[1])};
            mode.dataDimensions(arrayDims);

            // For now, hardcode azimuth and range looks until they exist in the product
            mode.numberAzimuthLooks(1);
            mode.numberRangeLooks(1);

            // Set PRF
            double value;
            isce::io::loadFromH5(modeGroup, "az_time_interval", value);
            mode.prf(1.0 / value);

            // Set bandwidth
            isce::io::loadFromH5(modeGroup, "bandwidth", value);
            mode.rangeBandwidth(value);

            // Set starting slant range
            isce::io::loadFromH5(modeGroup, "slant_range_start", value);
            mode.startingRange(value);

            // Set slant range pixel spacing
            isce::io::loadFromH5(modeGroup, "slant_range_spacing", value);
            mode.rangePixelSpacing(value);

            // Set wavelength
            isce::io::loadFromH5(modeGroup, "freq_center", value);
            mode.wavelength(isce::core::SPEED_OF_LIGHT / value);

            // Load zero doppler starting azimuth time
            isce::core::FixedString datestr;
            isce::io::loadFromH5(modeGroup, "zero_doppler_start_az_time", datestr);
            // Convert to datetime
            isce::core::DateTime date(std::string(datestr.str));
            // Set
            mode.startAzTime(date);

            // Load zero doppler ending azimuth time
            isce::io::loadFromH5(modeGroup, "zero_doppler_end_az_time", datestr);
            // Convert to datetime
            date = std::string(std::string(datestr.str));
            // Set
            mode.endAzTime(date);
        }

        /** Load ComplexImagery data from HDF5.
         *
         * @param[in] group         HDF5 group object.
         * @param[in] cpxImg        ComplexImagery object to be configured. */
        inline void loadFromH5(isce::io::IGroup & group, ComplexImagery & cpxImg) {

            // Configure a temporary auxiliary ImageMode
            ImageMode aux;
            loadFromH5(group, aux, "aux");
            // Save to cpxImg
            cpxImg.auxMode(aux);

            // Configure a temporary primary ImageMode
            ImageMode primary;
            loadFromH5(group, primary, "primary");
            // Save to cpxImg
            cpxImg.primaryMode(primary);
        }
        
    }
}

#endif

// end of file
