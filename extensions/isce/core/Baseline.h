//-*- C++ -*-
//-*- coding: utf-8 -*-

#ifndef ISCELIB_BASELINE_H
#define ISCELIB_BASELINE_H

// Standard
#include <vector>
#include <cmath>
#include <cstdio>

// isceLib
#include "Orbit.h"
#include "Metadata.h"
#include "Ellipsoid.h"
#include "Constants.h"
#include "LinAlg.h"

namespace isce { namespace core {

    typedef std::size_t size_t;

    class Baseline {

        public:
            orbitInterpMethod orbit_method;

            // Visible isce::core objects
            LinAlg linalg;
            Orbit orbit1, orbit2;
            RadarMetadata radar;
            Ellipsoid ellp;

            // Constructors
            Baseline();

            // Methods
            void init();
            void initBasis(double);
            std::vector<double> calculateBasisOffset(std::vector<double> &);
            void computeBaselines();

            // Utility methods
            double getHorizontalBaseline() {return _bh;}
            double getVerticalBaseline() {return _bv;}
            double getPerpendicularBaseline() {
                return (-1.0 * _bh * _coslook + _bv * _sinlook);
            }

        private:

            // Baseline scalars
            double _bh;
            double _bv;

            // Look vector components
            double _coslook, _sinlook;

            // Basis vectors
            std::vector<double> _refxyz = std::vector<double>(3);
            std::vector<double> _look = std::vector<double>(3);
            std::vector<double> _rhat = std::vector<double>(3);
            std::vector<double> _chat = std::vector<double>(3);
            std::vector<double> _vhat = std::vector<double>(3);

            // Velocity magnitude
            double _velocityMagnitude;

            // Private methods
            void _calculateLookVector(double);

    }; // class Baseline

} // namespace core
} // namespace isce

#endif

// end of file
