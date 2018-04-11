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
#include <isce/core/Metadata.h>
#include <isce/core/Poly2d.h>

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

    // Definition for Metadata
    template <class Archive>
    void serialize(Archive & archive, Metadata & meta) {
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
                cereal::make_nvp("antennaLength", meta.antennaLength));
    }

    // Definition for Poly2d
    template <class Archive>
    void serialize(Archive & archive, Poly2d & poly) {
        archive(cereal::make_nvp("rangeOrder", poly.rangeOrder),
                cereal::make_nvp("azimuthOrder", poly.azimuthOrder),
                cereal::make_nvp("rangeMean", poly.rangeMean),
                cereal::make_nvp("azimuthMean", poly.azimuthMean),
                cereal::make_nvp("rangeNorm", poly.rangeNorm),
                cereal::make_nvp("azimuthNorm", poly.azimuthNorm),
                cereal::make_nvp("coeffs", poly.coeffs));
    }

}}

#endif

// end of file
