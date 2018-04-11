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

namespace isce {
    namespace geometry {

        // Topo save does nothing
        template <class Archive>
        void save(Archive & archive, const Topo & topo) {}

        // Topo load
        template <class Archive>
        void load(Archive & archive, Topo & topo) {

            // Deserialize scalar values
            double threshold;
            size_t numiter, extraiter, orbitMethod, demMethod;
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
    }
}

#endif

// end of file
