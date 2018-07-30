// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Source Author: Bryan Riel
// Copyright 2017-2018

#ifndef ISCE_PRODUCT_IDENTIFICATION_H
#define ISCE_PRODUCT_IDENTIFICATION_H

// std
#include <string>

// pyre
#include <portinfo>
#include <pyre/journal.h>

// isce::core
#include <isce/core/Ellipsoid.h>

// Declarations
namespace isce {
    namespace product {
        class Identification;
    }
}

// Identification class declaration
class isce::product::Identification {

    public:
        /** Default constructor. */
        inline Identification() {};

        /** Copy constructor. */
        inline Identification(const Identification &);

        /** Assignment operator. */
        inline Identification & operator=(const Identification &);

        /** Get look direction */
        inline int lookDirection() const;
        /** Set look direction from integer. */
        inline void lookDirection(int);
        /** Set look direction from string. */
        inline void lookDirection(const std::string &);

        /** Get copy of ellipsoid */
        inline isce::core::Ellipsoid ellipsoid() const;
        /** Set ellipsoid */
        inline void ellipsoid(const isce::core::Ellipsoid &);
        
    private:
        // Ellipsoid
        isce::core::Ellipsoid _ellipsoid;

        // Look direction (-1 = right, +1 = left)
        int _lookDirection;

};

// Get inline implementations for Identification
#define ISCE_PRODUCT_IDENTIFICATION_ICC
#include "Identification.icc"
#undef ISCE_PRODUCT_IDENTIFICATION_ICC

#endif

// end of file
