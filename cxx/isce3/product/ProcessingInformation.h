//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2019

#pragma once

// std
#include <valarray>
#include <map>

// isce3::core
#include <isce3/core/LUT2d.h>

// Declaration
namespace isce3 {
    namespace product {
        class ProcessingInformation;
    }
}

// Definition of isce3::product::ProcessingInformation
class isce3::product::ProcessingInformation {

    public:
        /** Default constructor */
        inline ProcessingInformation() {}

        /** Copy constructor */
        inline ProcessingInformation(const ProcessingInformation & proc);

        /** Deep assignment operator */
        inline ProcessingInformation & operator=(const ProcessingInformation & proc);

        /** Get read-only look-up-table for effective velocity */
        inline const isce3::core::LUT2d<double> & effectiveVelocity() const {
            return _effectiveVelocity;
        }
        /** Set look-up-table for effective velocity */
        inline void effectiveVelocity(const isce3::core::LUT2d<double> & lut) {
            _effectiveVelocity = lut;
        }

        /** Get read-only map for azimuth FM rate tables */
        inline const std::map<char, isce3::core::LUT2d<double>> & azimuthFMRateMap() const {
            return _azimuthFMRate;
        }

        /** Get read-only look-up-table for azimuth FM rate */
        inline const isce3::core::LUT2d<double> & azimuthFMRate(char freq) const {
            return _azimuthFMRate.at(freq);
        }
        /** Set look-up-table for azimuth FM rate */
        inline void azimuthFMRate(const isce3::core::LUT2d<double> & lut, char freq) {
            _azimuthFMRate[freq] = lut;
        }

        /** Get read-only map for Doppler centroid tables */
        inline const std::map<char, isce3::core::LUT2d<double>> & dopplerCentroidMap() const {
            return _dopplerCentroid;
        }

        /** Get read-only look-up-table for Doppler centroid */
        inline const isce3::core::LUT2d<double> & dopplerCentroid(char freq) const {
            return _dopplerCentroid.at(freq);
        }
        /** Set look-up-table for azimuth FM rate */
        inline void dopplerCentroid(const isce3::core::LUT2d<double> & lut, char freq) {
            _dopplerCentroid[freq] = lut;
        }

    private:
        // Coordinates
        isce3::core::DateTime _refEpoch;

        // Constant look up tables
        isce3::core::LUT2d<double> _effectiveVelocity;

        // Frequency-dependent look up tables stored in maps
        std::map<char, isce3::core::LUT2d<double>> _azimuthFMRate;
        std::map<char, isce3::core::LUT2d<double>> _dopplerCentroid;
};

// Copy constructor
/** @param[in] proc ProcessingInformation */
isce3::product::ProcessingInformation::
ProcessingInformation(const isce3::product::ProcessingInformation & proc) {
    for (auto const & pair : proc.azimuthFMRateMap()) {
        _azimuthFMRate[pair.first] = pair.second;
    }
    for (auto const & pair: proc.dopplerCentroidMap()) {
        _dopplerCentroid[pair.first] = pair.second;
    }
}

// Deep assignment operator
/** @param[in] proc ProcessingInformation */
isce3::product::ProcessingInformation &
isce3::product::ProcessingInformation::
operator=(const isce3::product::ProcessingInformation & proc) {
    _effectiveVelocity = proc.effectiveVelocity();
    for (auto const & pair : proc.azimuthFMRateMap()) {
        _azimuthFMRate[pair.first] = pair.second;
    }
    for (auto const & pair: proc.dopplerCentroidMap()) {
        _dopplerCentroid[pair.first] = pair.second;
    }
    return *this;
}
