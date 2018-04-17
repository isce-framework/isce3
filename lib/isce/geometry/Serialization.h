//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018

#ifndef ISCE_GEOMETRY_SERIALIZATION_H
#define ISCE_GEOMETRY_SERIALIZATION_H

#include <iostream>
#include <memory>
#include <cereal/types/memory.hpp>
#include <cereal/archives/xml.hpp>

#include <isce/geometry/Topo.h>
#include <isce/geometry/Geo2rdr.h>

namespace isce {
    namespace geometry {

        // ----------------------------------------------------------------------
        // Serialization for Topo
        // ----------------------------------------------------------------------

        // Topo save does nothing
        template <class Archive>
        void save(Archive & archive, const Topo & topo) {}

        // Topo load
        template <class Archive>
        void load(Archive & archive, Topo & topo) {

            // Deserialize scalar values
            double threshold;
            size_t numiter, extraiter;
            isce::core::orbitInterpMethod orbitMethod;
            isce::core::dataInterpMethod demMethod;
            archive(cereal::make_nvp("threshold", threshold),
                    cereal::make_nvp("numIterations", numiter),
                    cereal::make_nvp("extraIterations", extraiter),
                    cereal::make_nvp("orbitMethod", orbitMethod),
                    cereal::make_nvp("demMethod", demMethod));

            // Send to Topo setters
            topo.threshold(threshold);
            topo.numiter(numiter);
            topo.extraiter(extraiter);
            topo.orbitMethod(orbitMethod);
            topo.demMethod(demMethod);
            topo.initialized(true);
        }

        // ----------------------------------------------------------------------
        // Serialization for Geo2rdr
        // ----------------------------------------------------------------------

        // Geo2rdr save does nothing
        template <class Archive>
        void save(Archive & archive, const Geo2rdr & geo) {}

        // Geo2rdr load
        template <class Archive>
        void load(Archive & archive, Geo2rdr & geo) {

            // Deserialize scalar values
            double threshold;
            size_t numiter;
            isce::core::orbitInterpMethod orbitMethod;
            archive(cereal::make_nvp("threshold", threshold),
                    cereal::make_nvp("numIterations", numiter),
                    cereal::make_nvp("orbitMethod", orbitMethod));

            // Send to Geo2rdr setters
            geo.threshold(threshold);
            geo.numiter(numiter);
            geo.orbitMethod(orbitMethod);
        }

    }
}

#endif

// end of file
