//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018

/** \file Serialization.h
 *
 * Serialization functions for isce::core objects. */

#ifndef ISCE_CORE_SERIALIZATION_H
#define ISCE_CORE_SERIALIZATION_H

#include <iostream>
#include <memory>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/xml.hpp>

// pyre
#include <portinfo>
#include <pyre/journal.h>

// isce::core
#include <isce/core/DateTime.h>
#include <isce/core/Ellipsoid.h>
#include <isce/core/Metadata.h>
#include <isce/core/Orbit.h>
#include <isce/core/Poly2d.h>
#include <isce/core/StateVector.h>

// isce::io
#include <isce/io/IH5.h>
#include <isce/io/Serialization.h>

//! The isce namespace
namespace isce {
    //! The isce::core namespace
    namespace core {

        // Archiving any isce::core object by pointer
        template <typename T>
        inline void load_archive(std::string metadata, char * objectTag, T * object) {
            std::stringstream metastream;
            metastream << metadata;
            cereal::XMLInputArchive archive(metastream);
            archive(cereal::make_nvp(objectTag, (*object)));
        }

        // Archiving any isce::core object by reference
        template <typename T>
        inline void load_archive_reference(std::string metadata, char * objectTag, T & object) {
            std::stringstream metastream;
            metastream << metadata;
            cereal::XMLInputArchive archive(metastream);
            archive(cereal::make_nvp(objectTag, object));
        }

        // ------------------------------------------------------------------------
        // Serialization for Ellipsoid
        // ------------------------------------------------------------------------

        template<class Archive>
        inline void save(Archive & archive, const Ellipsoid & ellps) {
            archive(cereal::make_nvp("a", ellps.a()),
                    cereal::make_nvp("e2", ellps.e2()));
        }

        template<class Archive>
        inline void load(Archive & archive, Ellipsoid & ellps) {
            double a, e2;
            archive(cereal::make_nvp("a", a),
                    cereal::make_nvp("e2", e2));
            ellps.a(a);
            ellps.e2(e2);
        }

        /** Load Ellipsoid parameters from HDF5.
         *
         * @param[in] file          HDF5 file object.
         * @param[in] ellps         Ellipsoid object to be configured. */
        inline void load(isce::io::IH5File & file, Ellipsoid & ellps) {
            // Find the ellipsoid dataset
            std::vector<std::string> ellpsList = file.find("ellipsoid");
            // Read data
            std::vector<double> ellpsData;
            isce::io::loadFromH5(file, ellpsList[0], ellpsData);
            // Set ellipsoid properties
            ellps.a(ellpsData[0]);
            ellps.e2(ellpsData[1]);
        }
        
        // ------------------------------------------------------------------------
        // Serialization for Orbit
        // ------------------------------------------------------------------------

        template <class Archive>
        inline void save(Archive & archive, const Orbit & orbit) {
            archive(cereal::make_nvp("StateVectors", orbit.stateVectors));
        }

        template <class Archive>
        inline void load(Archive & archive, Orbit & orbit) {
            // Load data
            archive(cereal::make_nvp("StateVectors", orbit.stateVectors));
            // Reformat state vectors to 1D arrays
            orbit.reformatOrbit();
        }

        /** \brief Load orbit data from HDF5 product.
         *
         * @param[in] file          HDF5 file object.
         * @param[in] orbit         Orbit object to be configured.
         * @param[in] orbit_type    orbit type (NOE, MOE, POE).
         * @param[in] refEpoch      DateTime reference epoch. */
        inline void load(isce::io::IH5File & file, Orbit & orbit, std::string orbit_type="POE",
                  DateTime refEpoch=MIN_DATE_TIME) {
            // Reset orbit data
            orbit.position.clear(); 
            orbit.velocity.clear(); 
            orbit.UTCtime.clear();
            orbit.epochs.clear();

            // Save the reference epoch
            orbit.refEpoch = refEpoch;

            // Load position
            isce::io::loadFromH5(file, "/science/metadata/orbit/" + orbit_type + "/position",
                                 orbit.position);

            // Load velocity
            isce::io::loadFromH5(file, "/science/metadata/orbit/" + orbit_type + "/velocity",
                                 orbit.velocity);

            // Load timestamp
            std::vector<FixedString> timestamps;
            isce::io::loadFromH5(file, "/science/metadata/orbit/" + orbit_type + "/timestamp",
                                 timestamps);
            orbit.nVectors = timestamps.size();
            orbit.UTCtime.resize(orbit.nVectors);
            orbit.epochs.resize(orbit.nVectors);

            // Finally, convert timestamps seconds
            for (int i = 0; i < orbit.nVectors; ++i) {
                // Make a string
                std::string timestampStr(timestamps[i].str);
                // Make a DateTime and save it
                DateTime date(timestampStr);
                orbit.epochs[i] = date;
                // Convert to seconds since epoch
                orbit.UTCtime[i] = date.secondsSinceEpoch(refEpoch);
            }
        }


        // ------------------------------------------------------------------------
        // Serialization for Metadata
        // ------------------------------------------------------------------------

        template <class Archive>
        inline void save(Archive & archive, const Metadata & meta) {
            archive(cereal::make_nvp("width", meta.width),
                    cereal::make_nvp("length", meta.length),
                    cereal::make_nvp("numberRangeLooks", meta.numberRangeLooks),
                    cereal::make_nvp("numberAzimuthLooks", meta.numberAzimuthLooks),
                    cereal::make_nvp("slantRangePixelSpacing", meta.slantRangePixelSpacing),
                    cereal::make_nvp("rangeFirstSample", meta.rangeFirstSample),
                    cereal::make_nvp("lookSide", meta.lookSide),
                    cereal::make_nvp("prf", meta.prf),
                    cereal::make_nvp("radarWavelength", meta.radarWavelength),
                    cereal::make_nvp("pegHeading", meta.pegHeading),
                    cereal::make_nvp("pegLatitude", meta.pegLatitude),
                    cereal::make_nvp("pegLongitude", meta.pegLongitude),
                    cereal::make_nvp("chirpSlope", meta.chirpSlope),
                    cereal::make_nvp("pulseDuration", meta.pulseDuration),
                    cereal::make_nvp("antennaLength", meta.antennaLength),
                    cereal::make_nvp("sensingStart", meta.sensingStart.isoformat()));
        }

        template <class Archive>
        inline void load(Archive & archive, Metadata & meta) {
            std::string sensingStart;
            archive(cereal::make_nvp("width", meta.width),
                    cereal::make_nvp("length", meta.length),
                    cereal::make_nvp("numberRangeLooks", meta.numberRangeLooks),
                    cereal::make_nvp("numberAzimuthLooks", meta.numberAzimuthLooks),
                    cereal::make_nvp("slantRangePixelSpacing", meta.slantRangePixelSpacing),
                    cereal::make_nvp("rangeFirstSample", meta.rangeFirstSample),
                    cereal::make_nvp("lookSide", meta.lookSide),
                    cereal::make_nvp("prf", meta.prf),
                    cereal::make_nvp("radarWavelength", meta.radarWavelength),
                    cereal::make_nvp("pegHeading", meta.pegHeading),
                    cereal::make_nvp("pegLatitude", meta.pegLatitude),
                    cereal::make_nvp("pegLongitude", meta.pegLongitude),
                    cereal::make_nvp("chirpSlope", meta.chirpSlope),
                    cereal::make_nvp("pulseDuration", meta.pulseDuration),
                    cereal::make_nvp("antennaLength", meta.antennaLength),
                    cereal::make_nvp("sensingStart", sensingStart));
            meta.sensingStart = sensingStart;
        }

        /** \brief Load radar metadata from HDF5 product.
         *
         * @param[in] file          HDF5 file object.
         * @param[in] meta          Metadata to be configured.
         * @param[in] mode          Imagery mode (aux, primary). */
        inline void load(isce::io::IH5File & file, Metadata & meta, std::string mode="primary") {

            double pri, bandwidth, centerFrequency;
            FixedString sensingStart, lookDir;

            // Make full mode string
            std::string path = "/science/complex_imagery/" + mode + "_mode";

            // Get dimension of imagery
            std::vector<int> dims = isce::io::getImageDims(file, path + "/hh");
            meta.length = dims[0];
            meta.width = dims[1];

            // Load values
            isce::io::loadFromH5(file, path + "/slant_range_start", meta.rangeFirstSample);
            isce::io::loadFromH5(file, path + "/slant_range_spacing", meta.slantRangePixelSpacing);
            isce::io::loadFromH5(file, path + "/az_time_interval", pri);
            isce::io::loadFromH5(file, path + "/bandwidth", bandwidth);
            isce::io::loadFromH5(file, path + "/pulse_duration", meta.pulseDuration);
            isce::io::loadFromH5(file, path + "/freq_center", centerFrequency);
            isce::io::loadFromH5(file, path + "/zero_doppler_start_az_time", sensingStart);
            isce::io::loadFromH5(file, "/science/metadata/identification/look_direction", lookDir);

            // Fields not currently defined in HDF5 product
            meta.numberRangeLooks = 1;
            meta.numberAzimuthLooks = 1;

            // Finalize rest of metadata items
            meta.prf = 1.0 / pri;
            meta.chirpSlope = bandwidth / meta.pulseDuration;
            meta.radarWavelength = SPEED_OF_LIGHT / centerFrequency;
            meta.sensingStart = std::string(sensingStart.str);
            std::string look(lookDir.str);
            if (look.compare("right") == 0) {
                meta.lookSide = -1;
            } else if (look.compare("left") == 0) {
                meta.lookSide = 1;
            }
        }

        // ------------------------------------------------------------------------
        // Serialization for Poly2d
        // ------------------------------------------------------------------------

        // Definition for Poly2d
        template <class Archive>
        inline void serialize(Archive & archive, Poly2d & poly) {
            archive(cereal::make_nvp("rangeOrder", poly.rangeOrder),
                    cereal::make_nvp("azimuthOrder", poly.azimuthOrder),
                    cereal::make_nvp("rangeMean", poly.rangeMean),
                    cereal::make_nvp("azimuthMean", poly.azimuthMean),
                    cereal::make_nvp("rangeNorm", poly.rangeNorm),
                    cereal::make_nvp("azimuthNorm", poly.azimuthNorm),
                    cereal::make_nvp("coeffs", poly.coeffs));
        }

        /** \brief Load polynomial coefficients from HDF5 product.
         *
         * @param[in] file          HDF5 file object.
         * @param[in] poly          Poly2d to be configured.
         * @param[in] name          Dataset name. */
        inline void load(isce::io::IH5File & file, Poly2d & poly, std::string name) {

            // Find the right Poly2d dataset
            std::vector<std::string> polys = file.find(name);
            if (polys.size() == 0) {
                pyre::journal::error_t errorChannel("isce.core.Serialization");
                errorChannel
                    << pyre::journal::at(__HERE__)
                    << "Could not find Poly2d dataset."
                    << pyre::journal::endl;
            }

            // Configure the polynomial coefficients
            isce::io::loadFromH5(file, polys[0], poly.coeffs);
            
            // Set other polynomial properties
            poly.rangeOrder = poly.coeffs.size() - 1;
            poly.azimuthOrder = 0;
            poly.rangeMean = 0.0;
            poly.azimuthMean = 0.0;
            poly.rangeNorm = 1.0;
            poly.azimuthNorm = 1.0;
        }

        // ------------------------------------------------------------------------
        // Serialization for StateVector
        // ------------------------------------------------------------------------

        // Serialization save method
        template<class Archive>
        inline void save(Archive & archive, StateVector const & sv) {
            // Archive
            archive(cereal::make_nvp("Time", sv.date().isoformat()),
                    cereal::make_nvp("Position", sv.positionToString()),
                    cereal::make_nvp("Velocity", sv.velocityToString()));
        }

        // Serialization load method
        template<class Archive>
        inline void load(Archive & archive, StateVector & sv) {
            // Make strings for position, velocity, and datetime
            std::string position_string, velocity_string, datetime_string;
            // Load the archive
            archive(cereal::make_nvp("Time", datetime_string),
                    cereal::make_nvp("Position", position_string),
                    cereal::make_nvp("Velocity", velocity_string));
            // Send position/velocity strings to parser
            sv.fromString(position_string, velocity_string);
            // Send datetime string to datetime object parser
            sv.date(datetime_string);
        }
    }
}

#endif

// end of file
