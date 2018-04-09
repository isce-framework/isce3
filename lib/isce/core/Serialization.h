//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2018

#ifndef ISCE_CORE_SERIALIZATION_H
#define ISCE_CORE_SERIALIZATION_H

#include <iostream>
#include <memory>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/xml.hpp>

#include <isce/core/Ellipsoid.h>

namespace isce { namespace core {

    // Main template call for archiving any isce::core object
    template <typename T>
    void load_archive(std::string metadata, char * objectTag, T * object) {
        std::stringstream metastream;
        metastream << metadata;
        cereal::XMLInputArchive archive(metastream);
        archive(cereal::make_nvp(objectTag, (*object)));
    }

    // Definition for Ellipsoid
    template<class Archive>
    void save(Archive & archive, const Ellipsoid & ellps) {
        archive(cereal::make_nvp("a", ellps.a()),
                cereal::make_nvp("e2", ellps.e2()));
    }

    template<class Archive>
    void load(Archive & archive, Ellipsoid & ellps) {
        double a, e2;
        archive(cereal::make_nvp("a", a),
                cereal::make_nvp("e2", e2));
        ellps.a(a);
        ellps.e2(e2);
    }

}}

#endif

// end of file
