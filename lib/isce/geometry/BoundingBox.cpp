// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Piyush Agram
// Copyright 2019

// isce::core
#include <isce/core/Constants.h>

// isce::geometry
#include "BoundingBox.h"

// pull in some isce::core namespaces
using isce::core::Vec3;
using isce::core::ProjectionBase;
using isce::core::Basis;

/** @param[in] proj ProjectionBase object indicating desired projection of output. 
 *
 * If input pointer is NULL, returns results in radians for lon, lat.
 * The outputs of this method are ready to be interpreted as a polygon shape in OGR/ shapely.
 * The sequence of walking the perimeter is always in the following order :
 * <ul>
 * <li> Start at Early Time, Near Range edge. Always the first point of the polygon.
 * <li> From there, Walk along the Early Time edge to Early Time, Far Range.
 * <li> From there, walk along the Far Range edge to Late Time, Far Range.
 * <li> From there, walk along the Late Time edge to Late Time, Near Range. 
 * <li> From there, walk along the Near Range edge back to Early Time, Near Range.
 * </ul>
 */
std::vector<Vec3> computePerimeter(ProjectionBase *proj)
{

    //Initialize journal
    pyre::journal::warning_t warning("isce.geometry.BoundingBox");

    //Initialize results
    std::vector<Vec3> perimeterPoints;

    //Skip factors along azimuth and range
    const double azSpacing = (_radarGrid.length() - 1.0) / (_pointsPerEdge - 1.0);
    const double rgSpacing = (_radarGrid.width() - 1.0) / (_pointsPerEdge - 1.0);

    //Store indices of image locations
    std::vector<double> azInd, rgInd;

    //Top Edge
    for (int ii = 0; ii < _pointsPerEdge; ii++)
    {
        azInd.push_back(0);
        rgInd.push_back( ii * rgSpacing );
    }

    //Right Edge
    for (int ii = 1; ii < _pointsPerEdge; ii++)
    {
        azInd.push_back( ii * azSpacing );
        rgInd.push_back( _radarGrid.width() - 1);
    }

    //Bottom Edge
    for (int ii = _pointsPerEdge-1; ii >= 0; ii--)
    {
        azInd.push_back( _radarGrid.length() - 1 );
        rgInd.push_back( ii * rgSpacing );
    }

    //Left Edge
    for (int ii = _pointsPerEdge-1; ii >= 0; ii--)
    {
        azInd.push_back( ii * azSpacing );
        rgInd.push_back(0);
    }

    //Create constant height DEM interpolator
    isce::geometry::DEMInterpolator constDEM(_terrainHeight);

    //Loop over indices
    for (int ii = 0; ii < rgInd.size(); ii++)
    {
        //Convert az index to azimuth time
        double tline = _radarGrid.sensingTime( azInd[ii] );

        //Initialize orbit data for this line
        Vec3 pos, vel;
        Basis TCNbasis;
        int status = _orbit.interpolate(tline, pos, vel, _orbitMethod);
        if (status != 0)
        {
            warning << pyre::journal::at(__HERE__)
                    << "Error in getting state vector for bounds computation."
                    << pyre::journal::newline
                    << " - requested time: " << tline << pyre::journal::newline
                    << " - bounds: " << _orbit.UTCtime[0] << "->" 
                    << _orbit.UTCTime[_orbit.nVectors-1]
                    << pyre::journal::endl;
            continue;
        }
           
        //Get geocentric TCN basis using satellite basis
        Basis TCNbasis = Basis(pos, vel);
        
        //Compute satellite velocity and height
        const double satVmag = vel.norm();
        const Vec3 satLLH = _ellipsoid.xyzToLonLat(pos);

        //Point to store solution
        Vec3 llh;

        //Get slant range and doppler factor
        double rng = _radarGrid.slantRange( rgInd[ii] );

        //If slant range vector doesn't hit ground, pick nadir point
        if (rng <= (satLLH[2] - _terrainHeight + 1.0))
        {
            llh = satLLH;
            llh[2] = _terrainHeight;
            perimeterPoints.push_back( llh );

            warning << "Possible near nadir imaging " << pyre::journal::endl;
            continue;
        }

        //Store in pixel object
        double dopfact = (0.5 * _radarGrid.wavelength() * (_doppler.eval(tline, rng)
                        /satVmag)) * rng;
        Pixel pixel(rng, dopfact, rbin);

        //Run rdr2geo
        rdr2geo(pixel, TCNbasis, pos, vel, _ellipsoid, constDEM, llh,
                _lookSide, _threshold, _numiter, 0);

        perimeterPoints.push_back(llh);
    }

    //If projection transformation is requested
    if (proj != nullptr)
    {
        Vec3 mapxyz;
        int status = 0;

        for(int ii=0; ii < perimeterPoints.size(); ii++)
        {
            int stat = proj->forward( perimeterPoints[ii], mapxyz);
            status = status || stat;

            perimeterPoints[ii] = mapxyz;
        }

        if (stat)
        {
            warning << pyre::journal::at(__HERE__)
                    << "Error in transforming perimeter to EPSG:"<< proj->code()
                    << pyre::journal::endl;
        }
    }

    //Check that atleast one point was transformed
    if (perimeterPoints.size() == 0)
    {
        warning << pyre::journal::at(__HERE__)
                << "Could not estimate any perimeter points"
                << pyre::journal::endl;
    }

    //Return points
    return perimeterPoints;
}
