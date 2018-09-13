// -*- coding: utf-8 -*-
//
// Source Author: Bryan Riel
// Copyright 2017-2018

#ifndef ISCE_CUDA_PRODUCT_GPUIMAGEMODE_H
#define ISCE_CUDA_PRODUCT_GPUIMAGEMODE_H

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_DEV __device__
#define CUDA_HOST __host__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#define CUDA_HOST
#endif

// isce::product
#include "isce/product/ImageMode.h"

// Declaration
namespace isce {
    namespace cuda {
        namespace product {
            class gpuImageMode;
        }
    }
}

// gpuImageMode class declaration
class isce::cuda::product::gpuImageMode {

    public:
        /** Default constructor. */
        CUDA_HOSTDEV inline gpuImageMode() {};

        /** Copy constructor on device. */
        CUDA_DEV inline gpuImageMode(const gpuImageMode &);

        /** Host-only copy constructor from ImageMode. */
        CUDA_HOST inline gpuImageMode(const isce::product::ImageMode &);

        /** Assignment operator. */
        CUDA_HOSTDEV inline gpuImageMode & operator=(const gpuImageMode &);

        /** Get length of image data. */
        CUDA_HOSTDEV inline size_t length() const { return _length; }

        /** Get width of image data. */
        CUDA_HOSTDEV inline size_t width() const { return _width; }

        /** Get pulse repetition frequency. */
        CUDA_HOSTDEV inline double prf() const { return _prf; }
        /** Set pulse repetition frequency. */
        CUDA_HOSTDEV inline void prf(double value) { _prf = value; }

        /** Get range bandwidth. */
        CUDA_HOSTDEV inline double rangeBandwidth() const { return _rangeBandwidth; }
        /** Set range bandwidth. */
        CUDA_HOSTDEV inline void rangeBandwidth(double value) { _rangeBandwidth = value; }

        /** Get radar wavelength. */
        CUDA_HOSTDEV inline double wavelength() const { return _wavelength; }
        /** Set radar wavelength. */
        CUDA_HOSTDEV inline void wavelength(double value) { _wavelength = value; }

        /** Get starting range. */
        CUDA_HOSTDEV inline double startingRange() const { return _startingRange; }
        /** Set starting range. */
        CUDA_HOSTDEV inline void startingRange(double value) { _startingRange = value; }

        /** Get range pixel spacing. */
        CUDA_HOSTDEV inline double rangePixelSpacing() const { return _rangePixelSpacing; }
        /** Set range pixel spacing. */
        CUDA_HOSTDEV inline void rangePixelSpacing(double value) { _rangePixelSpacing = value; }

        /** Get starting azimuth time in UTC seconds. */
        CUDA_HOSTDEV inline double startAzUTCTime() const { return _startAzUTCTime; }

        /** Get number of azimuth looks. */
        CUDA_HOSTDEV inline size_t numberAzimuthLooks() const { return _numberAzimuthLooks; }
        /** Set number of azimuth looks. */
        CUDA_HOSTDEV inline void numberAzimuthLooks(size_t value) { _numberAzimuthLooks = value; }

        /** Get number of range looks. */
        CUDA_HOSTDEV inline size_t numberRangeLooks() const { return _numberRangeLooks; }
        /** Set number of range looks. */
        CUDA_HOSTDEV inline void numberRangeLooks(size_t value) { _numberRangeLooks = value; }

    private:
        // Image related data
        size_t _length;
        size_t _width;

        // Instrument related data
        double _prf;
        double _rangeBandwidth;
        double _wavelength;
        double _startingRange;
        double _rangePixelSpacing;
        double _startAzUTCTime;

        // Looks
        size_t _numberAzimuthLooks;
        size_t _numberRangeLooks;
}; 

// Get inline implementation for gpuImageMode
#define ISCE_CUDA_PRODUCT_GPUIMAGEMODE_ICC
#include "gpuImageMode.icc"
#undef ISCE_CUDA_PRODUCT_GPUIMAGEMODE_ICC

#endif

// end of file
