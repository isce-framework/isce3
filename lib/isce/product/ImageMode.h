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
#include <vector>

// pyre
#include <portinfo>
#include <pyre/journal.h>

// isce::core
#include <isce/core/DateTime.h>
#include <isce/core/Metadata.h>

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
        inline ImageMode() : _modeType("primary"), _prf(1.0), _wavelength(1.0) {};

        /** Copy constructor. */
        inline ImageMode(const ImageMode &);

        /** Constructor with a specified mode type. */
        inline ImageMode(const std::string &);

        /** Constructor from isce::core::Metadata. */
        inline ImageMode(const isce::core::Metadata &);

        /** Assignment operator. */
        inline ImageMode & operator=(const ImageMode &);

        /** Get the path to the image data for a given polarization. */
        inline std::string dataPath(const std::string & pol);

        /** Get dimensions for image data. */
        inline std::array<size_t, 2> dataDimensions() const;
        /** Set dimensions for image data. */
        inline void dataDimensions(const std::array<size_t, 2> &);

        /** Get length of image data. */
        inline size_t length() const { return _imageDims[0]; }
        
        /** Get width of image data. */
        inline size_t width() const { return _imageDims[1]; }

        /** Get mode type ('aux' or 'primary'). */
        inline std::string modeType() const;
        /** Set mode type. */
        inline void modeType(const std::string &);

        /** Get pulse repetition frequency. */
        inline double prf() const { return _prf; }
        /** Set pulse repetition frequency. */
        inline void prf(double value) { _prf = value; }

        /** Get range bandwidth. */
        inline double rangeBandwidth() const { return _rangeBandwidth; }
        /** Set range bandwidth. */
        inline void rangeBandwidth(double value) { _rangeBandwidth = value; }

        /** Get radar wavelength. */
        inline double wavelength() const { return _wavelength; }
        /** Set radar wavelength. */
        inline void wavelength(double value) { _wavelength = value; }

        /** Get starting range. */
        inline double startingRange() const { return _startingRange; }
        /** Set starting range. */
        inline void startingRange(double value) { _startingRange = value; }

        /** Get range pixel spacing. */
        inline double rangePixelSpacing() const { return _rangePixelSpacing; }
        /** Set range pixel spacing. */
        inline void rangePixelSpacing(double value) { _rangePixelSpacing = value; }

        /** Get number of azimuth looks. */
        inline size_t numberAzimuthLooks() const { return _numberAzimuthLooks; }
        /** Set number of azimuth looks. */
        inline void numberAzimuthLooks(size_t value) { _numberAzimuthLooks = value; }

        /** Get number of range looks. */
        inline size_t numberRangeLooks() const { return _numberRangeLooks; }
        /** Set number of range looks. */
        inline void numberRangeLooks(size_t value) { _numberRangeLooks = value; }

        /** Get zero-doppler starting azimuth time. */
        inline isce::core::DateTime startAzTime() const { return _startAzTime; }
        /** Set zero-doppler starting azimuth time. */
        inline void startAzTime(const isce::core::DateTime & dtime) { _startAzTime = dtime; }

        /** Get zero-doppler ending azimuth time. */
        inline isce::core::DateTime endAzTime() const { return _endAzTime; }
        /** Set zero-doppler starting azimuth time. */
        inline void endAzTime(const isce::core::DateTime & dtime) { _endAzTime = dtime; }

    private:
        // Mode designation
        std::string _modeType;

        // Image related data
        std::array<size_t, 2> _imageDims;

        // Instrument related data
        double _prf;
        double _rangeBandwidth;
        double _wavelength;
        double _startingRange;
        double _rangePixelSpacing;

        // Looks
        size_t _numberAzimuthLooks;
        size_t _numberRangeLooks;

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
