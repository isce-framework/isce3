//-*- C++ -*-
//-*- coding: utf-8 -*-

#ifndef ISCE_PRODUCT_CONFIGPARAMETERS_H
#define ISCE_PRODUCT_CONFIGPARAMETERS_H

// isce::core
#include <isce/core/Metadata.h>

// isce::product
#include <isce/product/Product.h>

namespace isce {
    namespace product {
        class RadarGridParameters; 
    }
}

class isce::product::RadarGridParameters {

    public:
        /** Constructor with a product and number of looks. */
        inline RadarGridParameters(const isce::product::Product & product,
                                   char frequency = 'A',
                                   size_t numberAzimuthLooks = 1,
                                   size_t numberRangeLooks = 1);

        /** Constructor with a swath. */
        inline RadarGridParameters(const isce::product::Swath & swath,
                                   size_t numberAzimuthLooks = 1,
                                   size_t numberRangeLooks = 1);

        /** Constructor from an isce::core::Metadata object. */
        inline RadarGridParameters(const isce::core::Metadata & meta,
                                   const isce::core::DateTime & refEpoch,
                                   size_t numberAzimuthLooks = 1,
                                   size_t numberRangeLooks = 1);

        // Get data member values
        inline size_t numberAzimuthLooks() const { return _numberAzimuthLooks; }
        inline size_t numberRangeLooks() const { return _numberRangeLooks; }
        inline double sensingStart() const { return _sensingStart; }
        inline double wavelength() const { return _wavelength; }
        inline double prf() const { return _prf; }
        inline double startingRange() const { return _startingRange; }
        inline double rangePixelSpacing() const { return _rangePixelSpacing; }
        inline size_t length() const { return _rlength; }
        inline size_t width() const { return _rwidth; }
        inline double sensingStop() const {
            return _sensingStart + (_rlength - 1.0) / _prf;
        }

    // Protected data members can be accessed by derived classes
    protected:
        size_t _numberAzimuthLooks;
        size_t _numberRangeLooks;
        double _sensingStart;
        double _wavelength;
        double _prf;
        double _startingRange;
        double _rangePixelSpacing;
        size_t _rlength;
        size_t _rwidth;
};

// Constructor with a swath.
/** @param[in] swath Input swath
  * @param[in] numberAzimuthLooks Number of azimuth looks in input geometry
  * @param[in] numberRangeLooks Number of range looks in input geometry */
isce::product::RadarGridParameters::
RadarGridParameters(const isce::product::Swath & swath,
                    size_t numberAzimuthLooks,
                    size_t numberRangeLooks) :
    _numberAzimuthLooks(numberAzimuthLooks),
    _numberRangeLooks(numberRangeLooks),
    _sensingStart(swath.zeroDopplerTime()[0]),
    _wavelength(swath.processedWavelength()),
    _prf(swath.nominalAcquisitionPRF()),
    _startingRange(swath.slantRange()[0]),
    _rangePixelSpacing(swath.rangePixelSpacing()),
    _rlength(swath.lines()),
    _rwidth(swath.samples()) {}

// Constructor with a product
/** @param[in] product Input Product
  * @param[in] frequency Frequency designation
  * @param[in] numberAzimuthLooks Number of azimuth looks of input product
  * @param[in] numberRangeLooks Number of range looks of input product */
isce::product::RadarGridParameters::
RadarGridParameters(const isce::product::Product & product,
                    char frequency,
                    size_t numberAzimuthLooks,
                    size_t numberRangeLooks) :
    RadarGridParameters(product.swath(frequency), numberAzimuthLooks, numberRangeLooks) {}

// Constructor from an isce::core::Metadata object.
/** @param[in] meta isce::core::Metadata object
  * @param[in] refEpoch Reference epoch date
  * @param[in] numberAzimuthLooks Number of azimuth looks in input geometry
  * @param[in] numberRangeLooks Number of range looks in input geometry */
isce::product::RadarGridParameters::
RadarGridParameters(const isce::core::Metadata & meta,
                    const isce::core::DateTime & refEpoch,
                    size_t numberAzimuthLooks,
                    size_t numberRangeLooks) :
    _numberAzimuthLooks(numberAzimuthLooks),
    _numberRangeLooks(numberRangeLooks),
    _sensingStart((meta.sensingStart - refEpoch).getTotalSeconds()),
    _wavelength(meta.radarWavelength),
    _prf(meta.prf),
    _startingRange(meta.rangeFirstSample),
    _rangePixelSpacing(meta.slantRangePixelSpacing),
    _rlength(meta.length),
    _rwidth(meta.width) {}

#endif

// end of file
