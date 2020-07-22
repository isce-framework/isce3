//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018

#pragma once

#include <iostream>
#include <memory>
#include <cereal/types/memory.hpp>
#include <cereal/archives/xml.hpp>

#include <isce3/geometry/Topo.h>
#include <isce3/geometry/Geo2rdr.h>

namespace isce3 {
    namespace geometry {

        // Main template call for archiving any isce3::geometry object
        template <typename T>
        void load_archive(std::string metadata, char * objectTag, T * object) {
            std::stringstream metastream;
            metastream << metadata;
            cereal::XMLInputArchive archive(metastream);
            archive(cereal::make_nvp(objectTag, (*object)));
        }

        // ----------------------------------------------------------------------
        // Serialization for Topo
        // ----------------------------------------------------------------------

        // Topo save does nothing
        template <class Archive>
        void save(Archive &, const Topo &) {}

        // Topo load
        template <class Archive>
        void load(Archive & archive, Topo & topo) {

            // Deserialize scalar values
            double threshold;
            int epsgOut;
            size_t numiter, extraiter;
            isce3::core::dataInterpMethod demMethod;
            archive(cereal::make_nvp("threshold", threshold),
                    cereal::make_nvp("numIterations", numiter),
                    cereal::make_nvp("extraIterations", extraiter),
                    cereal::make_nvp("demMethod", demMethod),
                    cereal::make_nvp("epsgOut", epsgOut));

            // Send to Topo setters
            topo.threshold(threshold);
            topo.numiter(numiter);
            topo.extraiter(extraiter);
            topo.demMethod(demMethod);
            topo.epsgOut(epsgOut);
        }

        // ----------------------------------------------------------------------
        // Serialization for Geo2rdr
        // ----------------------------------------------------------------------

        // Geo2rdr save does nothing
        template <class Archive>
        void save(Archive &, const Geo2rdr &) {}

        // Geo2rdr load
        template <class Archive>
        void load(Archive & archive, Geo2rdr & geo) {

            // Deserialize scalar values
            double threshold;
            size_t numiter;
            archive(cereal::make_nvp("threshold", threshold),
                    cereal::make_nvp("numIterations", numiter));

            // Send to Geo2rdr setters
            geo.threshold(threshold);
            geo.numiter(numiter);
        }

    }
}
