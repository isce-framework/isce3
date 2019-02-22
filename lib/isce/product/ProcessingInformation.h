//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2019

#ifndef ISCE_PRODUCT_PROCESSINGINFORMATION_H
#define ISCE_PRODUCT_PROCESSINGINFORMATION_H

// std
#include <valarray>
#include <map>

// isce::core
#include <isce/core/LUT2d.h>

// Declaration
namespace isce {
    namespace product {
        class ProcessingInformation;
    }
}

// Definition of isce::product::ProcessingInformation
class isce::product::ProcessingInformation {

    public:
        /** Default constructor */
        inline ProcessingInformation() {}

        /** Copy constructor */
        inline ProcessingInformation(const ProcessingInformation & proc);

        /** Deep assignment operator */
        inline ProcessingInformation & operator=(const ProcessingInformation & proc);

        /** Get read-only slant range coordinates */
        inline const std::valarray<double> & slantRange() const { return _slantRange; }
        /** Set slant range coordinates */
        inline void slantRange(const std::valarray<double> & r) { _slantRange = r; }

        /** Get read-only zero Doppler time coordinates */
        inline const std::valarray<double> & zeroDopplerTime() const { return _zeroDopplerTime; }
        /** Set zero Doppler time coordinates */
        inline void zeroDopplerTime(const std::valarray<double> & t) { _zeroDopplerTime = t; }

        /** Get read-only look-up-table for effective velocity */
        inline const isce::core::LUT2d<double> & effectiveVelocity() const {
            return _effectiveVelocity;
        }
        /** Set look-up-table for effective velocity */
        inline void effectiveVelocity(const isce::core::LUT2d<double> & lut) {
            _effectiveVelocity = lut;
        }

        /** Get read-only map for azimuth FM rate tables */
        inline const std::map<char, isce::core::LUT2d<double>> & azimuthFMRateMap() const {
            return _azimuthFMRate;
        }

        /** Get read-only look-up-table for azimuth FM rate */
        inline const isce::core::LUT2d<double> & azimuthFMRate(char freq) const {
            return _azimuthFMRate.at(freq);
        }
        /** Set look-up-table for azimuth FM rate */
        inline void azimuthFMRate(const isce::core::LUT2d<double> & lut, char freq) {
            _azimuthFMRate[freq] = lut;
        }

        /** Get read-only map for Doppler centroid tables */
        inline const std::map<char, isce::core::LUT2d<double>> & dopplerCentroidMap() const {
            return _dopplerCentroid;
        }

        /** Get read-only look-up-table for Doppler centroid */
        inline const isce::core::LUT2d<double> & dopplerCentroid(char freq) const {
            return _dopplerCentroid.at(freq);
        }
        /** Set look-up-table for azimuth FM rate */
        inline void dopplerCentroid(const isce::core::LUT2d<double> & lut, char freq) {
            _dopplerCentroid[freq] = lut;
        }

    private:
        // Coordinates
        std::valarray<double> _slantRange;
        std::valarray<double> _zeroDopplerTime;
        isce::core::DateTime _refEpoch;

        // Constant look up tables
        isce::core::LUT2d<double> _effectiveVelocity;

        // Frequency-dependent look up tables stored in maps
        std::map<char, isce::core::LUT2d<double>> _azimuthFMRate;
        std::map<char, isce::core::LUT2d<double>> _dopplerCentroid;
};

// Copy constructor
/** @param[in] proc ProcessingInformation */
isce::product::ProcessingInformation::
ProcessingInformation(const isce::product::ProcessingInformation & proc) :
                      _slantRange(proc.slantRange()), _zeroDopplerTime(proc.zeroDopplerTime()) {
    for (auto const & pair : proc.azimuthFMRateMap()) {
        _azimuthFMRate[pair.first] = pair.second;
    }
    for (auto const & pair: proc.dopplerCentroidMap()) {
        _dopplerCentroid[pair.first] = pair.second;
    }
}

// Deep assignment operator
/** @param[in] proc ProcessingInformation */
isce::product::ProcessingInformation &
isce::product::ProcessingInformation::
operator=(const isce::product::ProcessingInformation & proc) {
    _slantRange = proc.slantRange();
    _zeroDopplerTime = proc.zeroDopplerTime();
    _effectiveVelocity = proc.effectiveVelocity();
    for (auto const & pair : proc.azimuthFMRateMap()) {
        _azimuthFMRate[pair.first] = pair.second;
    }
    for (auto const & pair: proc.dopplerCentroidMap()) {
        _dopplerCentroid[pair.first] = pair.second;
    }
    return *this;
}

#endif

// end of file
