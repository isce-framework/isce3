// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Source Author: Bryan Riel
// Copyright 2017-2018

#ifndef ISCE_PRODUCT_IMAGEMODE_H
#define ISCE_PRODUCT_IMAGEMODE_H

// std
#include <array>
#include <string>
#include <algorithm>

// pyre
#include <portinfo>
#include <pyre/journal.h>

// isce::core
#include <isce/core/DateTime.h>

// isce::io
#include <isce/io/IH5.h>

// Declaration
namespace isce {
    namespace product {
        class ImageMode;
    }
}

// ImageMode class declaration
class isce::product::ImageMode {

    public:
        /** Default constructor. */
        ImageMode() : _modeType("primary") {};

        /** Constructor with a specified mode type. */
        inline ImageMode(const std::string &);

        /** Assignment operator. */
        ImageMode & operator=(const ImageMode &);

        /** Get the path to the image data for a given polarization. */
        inline std::string dataPath(const std::string & pol);

        /** Get dimensions for image data for a given polarization. */
        inline std::array<size_t, 2> dataDimensions(const std::string & pol);

        /** Get mode type ('aux' or 'primary'). */
        inline std::string modeType() const;
        /** Set mode type. */
        inline void modeType(const std::string &);

        /** Get pulse repetition frequency. */
        double prf() const { return _prf; }
        /** Set pulse repetition frequency. */
        void prf(double value) { _prf = value; }

        /** Get range bandwidth. */
        double rangeBandwidth() const { return _rangeBandwidth; }
        /** Set range bandwidth. */
        void rangeBandwidth(double value) { _rangeBandwidth = value; }

        /** Get radar wavelength. */
        double wavelength() const { return _radarWavelength; }
        /** Set radar wavelength. */
        void wavelength(double value) { _radarWavelength = value; }

        /** Get pulse duration. */
        double pulseDuration() const { return _pulseDuration; }
        /** Set pulse duration. */
        void pulseDuration(double value) { _pulseDuration = value; }

        /** Get starting range. */
        double startingRange() const { return _startingRange; }
        /** Set starting range. */
        void startingRange(double value) { _startingRange = value; }

        /** Get range pixel spacing. */
        double rangePixelSpacing() const { return _rangePixelSpacing; }
        /** Set range pixel spacing. */
        void rangePixelSpacing(double value) { _rangePixelSpacing = value; }

        /** Get zero-doppler starting azimuth time. */
        isce::core::DateTime startAzTime() const { return _startAzTime; }
        /** Set zero-doppler starting azimuth time. */
        void startAzTime(const isce::core::DateTime & dtime) { _startAzTime = dtime; }

        /** Get zero-doppler ending azimuth time. */
        isce::core::DateTime endAzTime() const { return _endAzTime; }
        /** Set zero-doppler starting azimuth time. */
        void endAzTime(const isce::core::DateTime & dtime) { _endAzTime = dtime; }

    private:
        // Mode designation
        std::string _modeType;

        // Instrument related data
        double _prf;
        double _rangeBandwidth;
        double _wavelength;
        double _pulseDuration;
        double _startingRange;
        double _rangePixelSpacing;

        // Azimuth timing data
        isce::core::DateTime _startAzTime;
        isce::core::DateTime _endAzTime;
};

// Get inline implementations for ImageMode
#define ISCE_PRODUCT_IMAGEMODE_ICC
#include "ImageMode.icc"
#undef ISCE_PRODUCT_IMAGEMODE_ICC

#endif

// end of file
