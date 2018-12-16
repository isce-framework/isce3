// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018
//

#ifndef ISCE_RADAR_RADAR_H
#define ISCE_RADAR_RADAR_H

// std
#include <valarray>

// isce::core
#include <isce/core/LUT1d.h>

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

        /** Constructor with slant range coordinates and Doppler values. */
        inline Radar(const std::valarray<double> & slantRange,
                     const std::valarray<double> & skew,
                     const std::valarray<double> & content);

        /** Constructor with pre-constructed skew and content Doppler LUTs. */
        inline Radar(const isce::core::LUT1d<double> & skew,
                     const isce::core::LUT1d<double> & content);

        /** Copy constructor. */
        inline Radar(const Radar &);

        /** Assignment operator. */
        inline Radar & operator=(const Radar &);

        /** Get copy of content Doppler LUT */
        inline isce::core::LUT1d<double> contentDoppler() const;
        /** Set content Doppler polynomial */
        inline void contentDoppler(const isce::core::LUT1d<double> &);

        /** Get copy of skew Doppler LUT */
        inline isce::core::LUT1d<double> skewDoppler() const;
        /** Set skew Doppler polynomial */
        inline void skewDoppler(const isce::core::LUT1d<double> &);
        
    private:
        // Doppler data
        isce::core::LUT1d<double> _skewDoppler;
        isce::core::LUT1d<double> _contentDoppler;

};

// Get inline implementations for Radar
#define ISCE_RADAR_RADAR_ICC
#include "Radar.icc"
#undef ISCE_RADAR_RADAR_ICC

#endif

// end of file
