// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Source Author: Bryan Riel
// Copyright 2017-2019

#ifndef ISCE_PRODUCT_METADATA_H
#define ISCE_PRODUCT_METADATA_H

// isce::core
#include <isce/core/EulerAngles.h>
#include <isce/core/Orbit.h>

// isce::product
#include <isce/product/ProcessingInformation.h>

// Declarations
namespace isce {
    namespace product {
        class Metadata;
    }
}

// Declare Metadata class
class isce::product::Metadata {

    public:
        /** Default constructor */
        inline Metadata() {}

        /** Copy constructor */
        inline Metadata(const Metadata &);

        /** Get read-only reference to attitude */
        inline const isce::core::EulerAngles & attitude() const { return _attitude; }
    
        /** Get reference to attitude */
        inline isce::core::EulerAngles & attitude() { return _attitude; }

        /** Set attitude */
        inline void attitude(const isce::core::EulerAngles & att) { _attitude = att; }

        /** Get read-only reference to orbit */
        inline const isce::core::Orbit & orbit() const { return _orbit; };

        /** Get reference to orbit */
        inline isce::core::Orbit & orbit() { return _orbit; }

        /** Set orbit */
        inline void orbit(const isce::core::Orbit & orb) { _orbit = orb; };

        /** Get read-only reference to ProcessingInformation */
        inline const ProcessingInformation & procInfo() const { return _procInfo; }

        /** Get reference to ProcessingInformation */
        inline ProcessingInformation & procInfo() { return _procInfo; }
        
    private:
        // Attitude
        isce::core::EulerAngles _attitude;
        // Orbit
        isce::core::Orbit _orbit;
        // ProcessingInformation
        isce::product::ProcessingInformation _procInfo;
};

// Copy constructor
/** @param[in] meta Metadata object */
isce::product::Metadata::
Metadata(const Metadata & meta) : _attitude(meta.attitude()), _orbit(meta.orbit()),
                                  _procInfo(meta.procInfo()) {}

#endif

// end of file
