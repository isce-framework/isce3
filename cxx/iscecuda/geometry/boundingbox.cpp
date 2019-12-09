// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Piyush Agram
// Copyright 2019

// header file
#include "boundingbox.h"

// pyre::journal
#include <pyre/journal.h>

// isce::core
#include <isce/core/Constants.h>
#include <isce/core/Basis.h>
#include <isce/core/Pixel.h>
#include <isce/core/Projections.h>

// isce::except
#include <isce/except/Error.h>

// isce::geometry
#include "DEMInterpolator.h"

// pull in some isce::core namespaces
using isce::core::Vec3;
using isce::core::ProjectionBase;
using isce::core::Basis;

/** @param[in] radarGrid    RadarGridParameters object
 * @param[in] orbit         Orbit object
 * @param[in] proj          ProjectionBase object indicating desired projection of output. 
 * @param[in] doppler       LUT2d doppler model
 * @param[in] demInterp     DEM Interpolator
 * @param[in] pointsPerEge  Number of points to use on each edge of radar grid
 * @param[in] threshold     Slant range threshold for convergence 
 * @param[in] numiter       Max number of iterations for convergence
 *
 * The outputs of this method is an OGRLinearRing.
 */
isce::geometry::Perimeter
isce::geometry::
getGeoPerimeter(const isce::product::RadarGridParameters &radarGrid,
                const isce::core::Orbit &orbit,
                const isce::core::ProjectionBase *proj,
                const isce::core::LUT2d<double> &doppler,
                const isce::geometry::DEMInterpolator &demInterp,
                const int pointsPerEdge,
                const double threshold,
                const int numiter)
{

    //Journal for warning
    pyre::journal::warning_t warning("isce.geometry.perimeter");

    //Initialize results
    isce::geometry::Perimeter perimeter;

    //Skip factors along azimuth and range
    const double azSpacing = (radarGrid.length() - 1.0) / (pointsPerEdge - 1.0);
    const double rgSpacing = (radarGrid.width() - 1.0) / (pointsPerEdge - 1.0);

    //Store indices of image locations
    //This could potentially be moved to RadarGridParamters.perimeter()
    //But that would introduce new dependency on shapes.h for RadarGridParameters
    std::vector<double> azInd, rgInd;

    //Top Edge
    for (int ii = 0; ii < pointsPerEdge; ii++)
    {
        azInd.push_back(0);
        rgInd.push_back( ii * rgSpacing );
    }

    //Right Edge
    for (int ii = 1; ii < pointsPerEdge; ii++)
    {
        azInd.push_back( ii * azSpacing );
        rgInd.push_back( radarGrid.width() - 1);
    }

    //Bottom Edge
    for (int ii = pointsPerEdge-2; ii >= 0; ii--)
    {
        azInd.push_back( radarGrid.length() - 1 );
        rgInd.push_back( ii * rgSpacing );
    }

    //Left Edge
    for (int ii = pointsPerEdge-2; ii >= 0; ii--)
    {
        azInd.push_back( ii * azSpacing );
        rgInd.push_back(0);
    }

    //Loop over indices
    for (int ii = 0; ii < rgInd.size(); ii++)
    {
        //Convert az index to azimuth time
        double tline = radarGrid.sensingTime( azInd[ii] );

        //Initialize orbit data for this line
        Vec3 pos, vel;
        orbit.interpolate(&pos, &vel, tline, 
                          isce::core::OrbitInterpBorderMode::Error);
           
        //Get geocentric TCN basis using satellite basis
        Basis tcnbasis(pos, vel);
        
        //Compute satellite velocity and height
        const double satVmag = vel.norm();
        const Vec3 satLLH = proj->ellipsoid().xyzToLonLat(pos);

        //Point to store solution
        Vec3 llh, mapxyz;

        //Get slant range and doppler factor
        double rng = radarGrid.slantRange( rgInd[ii] );

        //If slant range vector doesn't hit ground, pick nadir point
        double nadirHgt = demInterp.interpolateLonLat(satLLH[0], satLLH[1]); 
        if (rng <= (satLLH[2] - nadirHgt + 1.0))
        {
            llh = satLLH;
            llh[2] = nadirHgt;

            warning << "Possible near nadir imaging " << pyre::journal::endl;
        }

        //Store in pixel object
        //Compute Doppler Factor related to rate of slant range change
        double dopplerFactor = (0.5 * radarGrid.wavelength() * (doppler.eval(tline, rng)
                        /satVmag)) * rng;
        isce::core::Pixel pixel(rng, dopplerFactor, rgInd[ii]);

        //Run rdr2geo
        rdr2geo(pixel, tcnbasis, pos, vel, proj->ellipsoid(), demInterp, llh,
                radarGrid.lookSide(), threshold, numiter, 0);

        //Transform point to projection
        int status = proj->forward(llh, mapxyz);
        if (status)
        {
            std::string errstr = "Error in transforming point (" + std::to_string(llh[0]) +
                                 "," + std::to_string(llh[1]) + "," + std::to_string(llh[2]) +
                                 ") to projection EPSG:" + std::to_string(proj->code());
            throw isce::except::OutOfRange(ISCE_SRCINFO(), errstr); 
        }

        //Add transformed point to the perimeter
        perimeter.addPoint(mapxyz[0], mapxyz[1], mapxyz[2]);

    }

    //Ensure polygon is closed
    perimeter.closeRings();

    //Return points
    return perimeter;
}


/** @param[in] radarGrid    RadarGridParameters object
 * @param[in] orbit         Orbit object
 * @param[in] proj          ProjectionBase object indicating desired projection of output. 
 * @param[in] doppler       LUT2d doppler model
 * @param[in] hgts          Vector of heights to use for the scene
 * @param[in] margin        Marging to add to estimated bounding box in decimal degrees
 * @param[in] pointsPerEge  Number of points to use on each edge of radar grid
 * @param[in] threshold     Slant range threshold for convergence 
 * @param[in] numiter       Max number of iterations for convergence
 */
isce::geometry::BoundingBox
isce::geometry::
getGeoBoundingBox(const isce::product::RadarGridParameters &radarGrid,
                  const isce::core::Orbit &orbit,
                  const isce::core::ProjectionBase *proj,
                  const isce::core::LUT2d<double> &doppler,
                  const std::vector<double> &hgts,
                  const double margin,
                  const int pointsPerEdge,
                  const double threshold,
                  const int numiter)
{
    //Initialize data structure for final output
    isce::geometry::BoundingBox bbox;

    //Loop over the heights
    for (const auto & height: hgts)
    {
        //Get perimeter for constant height
        isce::geometry::DEMInterpolator constDEM(height);
        isce::geometry::Perimeter perimeter = getGeoPerimeter(radarGrid, orbit,
                                                  proj, doppler, constDEM,
                                                  pointsPerEdge, threshold,
                                                  numiter);
        
        //Get bounding box for given height
        isce::geometry::BoundingBox xylim;
        perimeter.getEnvelope(&xylim);
    
        //If lat/lon coordinates need to be adjusted before estimating limits
        if ((proj->code() == 4326) && ((xylim.MaxX - xylim.MinX) > 180.0))
        {
            OGRPoint pt;
            for(int ii=0; ii < perimeter.getNumPoints(); ii++)
            {
                perimeter.getPoint(ii, &pt);
                double X = pt.getX();
                if (X < 0.)
                    pt.setX(X+360.0);

                perimeter.setPoint(ii, &pt);
            }
            //Re-estimate limits with adjusted longitudes
            perimeter.getEnvelope(&xylim);
        }

        //Merge with other bboxes
        bbox.Merge(xylim);
    }

    //Set up margin in meters / degrees
    double delta = margin; 
    if (proj->code() != 4326)
        delta = isce::core::decimaldeg2meters(margin);

    bbox.MinX -= delta;
    bbox.MaxX += delta;
    bbox.MinY -= delta;
    bbox.MaxX += delta;

    //Special checks for lonlat
    if (proj->code() == 4326)
    {
        //If there is a dateline crossing
        if ((bbox.MaxX-bbox.MinX) > 180.0) 
        {
            double maxx = bbox.MinX + 360.0;
            bbox.MinX = bbox.MaxX;
            bbox.MaxX = maxx;
        }

        //Check for north pole 
        bbox.MaxY = std::min(bbox.MaxY, 90.0);

        //Check for south pole
        bbox.MinY = std::max(bbox.MinY, -90.0);

    }

    //Return the estimated bounding box
    return bbox;
}

//end of file
