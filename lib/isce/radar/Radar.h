// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018
//

#ifndef ISCE_RADAR_RADAR_H
#define ISCE_RADAR_RADAR_H

// isce::core
#include "isce/core/Poly2d.h"

// Declaration
namespace isce {
    namespace radar {
        class Radar;
    }
}

// Radar class declaration
class isce::radar::Radar {

    public:
        /** Default constructor. */
        inline Radar() {};

        /** Constructor with skew and content Dopplers. */
        inline Radar(const isce::core::Poly2d & skew, const isce::core::Poly2d & content);

        /** Copy constructor. */
        inline Radar(const Radar &);

        /** Assignment operator. */
        inline Radar & operator=(const Radar &);

        /** Get copy of content Doppler polynomial */
        inline isce::core::Poly2d contentDoppler() const;
        /** Set content Doppler polynomial */
        inline void contentDoppler(const isce::core::Poly2d &);

        /** Get copy of skew Doppler polynomial */
        inline isce::core::Poly2d skewDoppler() const;
        /** Set skew Doppler polynomial */
        inline void skewDoppler(const isce::core::Poly2d &);
        
    private:
        // Doppler data
        isce::core::Poly2d _skewDoppler;
        isce::core::Poly2d _contentDoppler;
};

// Get inline implementations for Radar
#define ISCE_RADAR_RADAR_ICC
#include "Radar.icc"
#undef ISCE_RADAR_RADAR_ICC

#endif

// end of file
