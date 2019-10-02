//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Piyush Agram
// Copyright 2019

#ifndef ISCE_GEOMETRY_BOUNDINGBOX_H
#define ISCE_GEOMETRY_BOUNDINGBOX_H

//pyre
#include <pyre/journal.h>

//isce::product
#include <isce/product/RadarGridParameters.h>

//isce::geometry
#include "geometry.h"

//Declaration
namespace isce{
    namespace geometry{
        class Rdr2GeoBoundingBox;
        //class Geo2RdrBoundingBox;
    }
}

/** Transformer from radar geometry coordinates to map coordiantes with reference altitude
 */
class isce::geometry::Rdr2GeoBoundingBox {

    public:
        /** Constructor using a radar grid*/
        inline Rdr2GeoBoundingBox(const isce::product::RadarGridParameters & radarGrid,
                                  const isce::core::Orbit & orbit,
                                  const isce::core::Ellipsoid & ellipsoid,
                                  const int lookSide,
                                  const isce::core::LUT2d<double> & doppler = isce::core::LUT2d<double>());

        /** Set convergence threshold */
        inline void threshold(double);

        /** Set number of iterations */
        inline void numiter(int);

        /** Set orbit interpolation Method */
        inline void orbitMethod(isce::core::orbitInterpMethod);

        /** Set reference height above ellipsoid */
        inline void terrainHeight(double);

        /** Set number of points along an edge*/
        inline void pointsPerEdge(int);

        /** Return number of points in a valid bounding polygon if all points transformed correctly without errors */
        inline int numPerimeterPoints();

        /** Return estimated bounding box */
        std::vector<isce::core::Vec3>
        computePerimeter(isce::core::ProjectionBase *proj);

        /** Return bounding box limits with margin */
        isce::core::Vector<4>
        computeBounds(isce::core::ProjectionBase *proj, double margin = 0.0);

    private:
        //isce::core::Objects
        isce::core::Orbit _orbit;
        isce::core::LUT2d<double> _doppler;

        //Radar grid
        isce::product::RadarGridParameters _radarGrid;

        //Geometry options
        int _lookSide;
        double _terrainHeight;
        isce::core::orbitInterpMethod _orbitMethod;

        //Optimization options
        int _pointsPerEdge = 11;
        int _numiter = 15;
        int _threshold = 1.0e-8;

};

//Get inline implementation for Bounding Box
#define ISCE_GEOMETRY_BOUNDINGBOX_ICC
#include "BoundingBox.icc"
#undef ISCE_GEOMETRY_BOUNDINGBOX_ICC

#endif

//end of file
