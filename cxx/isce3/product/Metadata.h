// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Source Author: Bryan Riel
// Copyright 2017-2019

#pragma once

// isce3::core
#include <isce3/core/Attitude.h>
#include <isce3/core/Orbit.h>

// isce3::product
#include <isce3/product/ProcessingInformation.h>

// Declarations
namespace isce3 {
    namespace product {
        class Metadata;
    }
}

// Declare Metadata class
class isce3::product::Metadata {

    public:
        /** Default constructor */
        inline Metadata() {}

        /** Copy constructor */
        inline Metadata(const Metadata &);

        /** Get read-only reference to attitude */
        const isce3::core::Attitude & attitude() const { return _attitude; }

        /** Get reference to attitude */
        inline isce3::core::Attitude & attitude() { return _attitude; }

        /** Set attitude */
        inline void attitude(const isce3::core::Attitude & att) { _attitude = att; }

        /** Get read-only reference to orbit */
        inline const isce3::core::Orbit & orbit() const { return _orbit; };

        /** Get reference to orbit */
        inline isce3::core::Orbit & orbit() { return _orbit; }

        /** Set orbit */
        inline void orbit(const isce3::core::Orbit & orb) { _orbit = orb; };

        /** Get read-only reference to ProcessingInformation */
        inline const ProcessingInformation & procInfo() const { return _procInfo; }

        /** Get reference to ProcessingInformation */
        inline ProcessingInformation & procInfo() { return _procInfo; }
        
    private:
        // Attitude
        isce3::core::Attitude _attitude;
        // Orbit
        isce3::core::Orbit _orbit;
        // ProcessingInformation
        isce3::product::ProcessingInformation _procInfo;
};

// Copy constructor
/** @param[in] meta Metadata object */
isce3::product::Metadata::
Metadata(const Metadata & meta) : _attitude(meta.attitude()), _orbit(meta.orbit()),
                                  _procInfo(meta.procInfo()) {}
