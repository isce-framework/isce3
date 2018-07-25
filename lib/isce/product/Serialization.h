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

// isce::product
#include <isce/product/ComplexImagery.h>
#include <isce/product/ImageMode.h>
#include <isce/product/Identification.h>
#include <isce/product/Metadata.h>

// isce::io
#include <isce/io/IH5.h>
#include <isce/io/Serialization.h>

//! The isce namespace
namespace isce {
    //! The isce::product namespace
    namespace product {

        /** Load top level Product from HDF5.
         *
         * @param[in] file          HDF5 file object.
         * @param[in] product       Product object to be configured. */
        void load(isce::io::IH5File & file, Product & product) {
            // Configure complex imagery
            load(file, product.complexImagery());
            // Configure metadata
            load(file, product.metadata());
        }

        /** Load ComplexImagery data from HDF5.
         *
         * @param[in] file          HDF5 file object.
         * @param[in] cpxImg        ComplexImagery object to be configured. */
        void load(isce::io::IH5File & file, ComplexImagery & cpxImg) {

            // Configure a temporary auxiliary ImageMode
            ImageMode aux;
            load(file, aux, "aux");
            // Save to cpxImg
            cpxImg.auxMode(aux);

            // Configure a temporary primary ImageMode
            ImageMode primary;
            load(file, primary, "primary");
            // Save to cpxImg
            cpxImg.primaryMode(primary);
        }

        /** Load ImageMode data from HDF5.
         *
         * @param[in] file          HDF5 file object.
         * @param[in] mode          ImageMode object to be configured. 
         * @param[in] modeType      String representing mode type. */
        void load(isce::io::IH5File & file, ImageMode & mode, std::string & modeType) {

            // Make path for mode
            std::string path = "/science/complex_imagery/" + modeType + "_mode";
            // Set mode type
            mode.modeType(modeType);

            // Set PRF
            double value;
            isce::io::loadFromH5(file, path + "/az_time_interval", value);
            mode.prf(1.0 / value);

            // Set bandwidth
            isce::io::loadFromH5(file, path + "/bandwidth", value);
            mode.rangeBandwidth(value);

            // Set pulse duration
            isce::io::loadFromH5(file, path + "/pulse_duration", value);
            mode.pulseDuration(value);

            // Set wavelength
            isce::io::loadFromH5(file, path + "/freq_center", value);
            mode.wavelength(isce::core::SPEED_OF_LIGHT / value);

            // Load zero doppler starting azimuth time
            isce::core::FixedString datestr;
            isce::io::loadFromH5(file, path + "/zero_doppler_start_az_time", datestr);
            // Convert to datetime
            isce::core::DateTime date(std::string(datestr.str));
            // Set
            mode.startAzTime(date);

            // Load zero doppler ending azimuth time
            isce::io::loadFromH5(file, path + "/zero_doppler_end_az_time", datestr);
            // Convert to datetime
            date = std::string(datestr.str);
            // Set
            mode.endAzTime(date);
        }

        /** Load Metadata parameters from HDF5.
         *
         * @param[in] file          HDF5 file object.
         * @param[in] meta          Metadata object to be configured. */
        void load(isce::io::IH5File & file, Metadata & meta) {

            // Configure a temporary orbit with NOE data
            isce::core::Orbit orbit;
            isce::core::load(file, orbit, "NOE");
            // Save to metadata
            meta.orbitNOE(orbit);

            // Configure a temporary orbit with POE data
            isce::core::load(file, orbit, "POE");
            // Save to metadata
            meta.orbitPOE(orbit);

            // Configure a temporary instrument
            isce::radar::Radar instrument;
            isce::radar::load(file, instrument);
            // Save to metadata
            meta.instrument(instrument);

            // Configure a temporary identification
            Identification id;
            load(file, id);
            // Save to metadata
            meta.identification(id);
        }

        /** Load Identification parameters from HDF5.
         *
         * @param[in] file      HDF5 file object.
         * @param[in] id        Identification object to be configured. */
        void load(isce::io::IH5File & file, Identification & id) {

            // Configure a temporary ellipsoid
            isce::core::Ellipsoid ellps;
            isce::core::load(file, ellps);
            // Save to identification
            id.ellipsoid(ellps);

            // Load the look direction
            std::string lookDir;
            isce::io::loadFromH5(
                file,
                "/science/metadata/identification/look_direction",
                lookDir
            );
            // Save to identification
            id.lookDirection(lookDir);
        }
        
    }
}

#endif

// end of file
