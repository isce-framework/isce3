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
#include <isce3/core/Basis.h>
#include <isce3/core/Constants.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Pixel.h>
#include <isce3/core/Projections.h>

// isce::geometry
#include <isce3/geometry/geometry.h>

// isce::except
#include <isce3/except/Error.h>

// isce::geometry
#include "DEMInterpolator.h"

// pull in some isce::core namespaces
using isce::core::Vec3;
using isce::core::ProjectionBase;
using isce::core::Basis;

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

    //Check for number of points on edge
    if (pointsPerEdge <= 2)
    {
        std::string errstr = "At least 2 points per edge should be requested "
                             "for perimeter estimation. " +
                             std::to_string(pointsPerEdge) + " requested. ";
        throw isce::except::OutOfRange(ISCE_SRCINFO(), errstr); 
    }

    //Journal for warning
    pyre::journal::warning_t warning("isce.geometry.perimeter");

    //Initialize results
    isce::geometry::Perimeter perimeter;

    //Ellipsoid being used for processing
    const isce::core::Ellipsoid &ellipsoid = proj->ellipsoid();

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

        //Get rg index to slant range
        double rng = radarGrid.slantRange( rgInd[ii] );

        //Get doppler at pixel of interest
        double dopp = doppler.eval(tline, rng);

        //Target location
        Vec3 llh, mapxyz;

        //Run rdr2geo
        rdr2geo(tline, rng, dopp,
                orbit, ellipsoid, demInterp, llh,
                radarGrid.wavelength(), radarGrid.lookSide(),
                threshold, numiter, 0); 

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

static void _addMarginToBoundingBox(isce::geometry::BoundingBox& bbox,
                                    const double margin,
                                    const isce::core::ProjectionBase* proj) {

    // Set up margin in meters / degrees
    double delta = margin;
    if (proj->code() != 4326)
        delta = isce::core::decimaldeg2meters(margin);

    bbox.MinX -= delta;
    bbox.MaxX += delta;
    bbox.MinY -= delta;
    bbox.MaxX += delta;

    // Special checks for lonlat
    if (proj->code() == 4326) {
        // If there is a dateline crossing
        if ((bbox.MaxX - bbox.MinX) > 180.0) {
            double maxx = bbox.MinX + 360.0;
            bbox.MinX = bbox.MaxX;
            bbox.MaxX = maxx;
        }

        // Check for north pole
        bbox.MaxY = std::min(bbox.MaxY, 90.0);

        // Check for south pole
        bbox.MinY = std::max(bbox.MinY, -90.0);
    }
}

isce::geometry::BoundingBox isce::geometry::getGeoBoundingBox(
        const isce::product::RadarGridParameters& radarGrid,
        const isce::core::Orbit& orbit, const isce::core::ProjectionBase* proj,
        const isce::core::LUT2d<double>& doppler,
        const std::vector<double>& hgts, const double margin,
        const int pointsPerEdge, const double threshold, const int numiter,
        bool ignore_out_of_range_exception) {

    // Check for number of points on edge
    if (margin < 0.) {
        std::string errstr = "Margin should be a positive number. " +
                             std::to_string(margin) + " requested. ";
        throw isce::except::OutOfRange(ISCE_SRCINFO(), errstr);
    }

    // Initialize data structure for final output
    isce::geometry::BoundingBox bbox;

    // Loop over the heights
    for (const auto& height : hgts) {
        // Get perimeter for constant height
        isce::geometry::DEMInterpolator constDEM(height);
        isce::geometry::Perimeter perimeter;

        if (ignore_out_of_range_exception) {
            try {
                perimeter = getGeoPerimeter(radarGrid, orbit, proj, doppler,
                                            constDEM, pointsPerEdge, threshold,
                                            numiter);
            } catch (isce::except::OutOfRange) {
                continue;
            }
        } else {
            perimeter =
                    getGeoPerimeter(radarGrid, orbit, proj, doppler, constDEM,
                                    pointsPerEdge, threshold, numiter);
        }

        // Get bounding box for given height
        isce::geometry::BoundingBox xylim;
        perimeter.getEnvelope(&xylim);

        // If lat/lon coordinates need to be adjusted before estimating limits
        if ((proj->code() == 4326) && ((xylim.MaxX - xylim.MinX) > 180.0)) {
            OGRPoint pt;
            for (int ii = 0; ii < perimeter.getNumPoints(); ii++) {
                perimeter.getPoint(ii, &pt);
                double X = pt.getX();
                if (X < 0.)
                    pt.setX(X + 360.0);

                perimeter.setPoint(ii, &pt);
            }
            // Re-estimate limits with adjusted longitudes
            perimeter.getEnvelope(&xylim);
        }

        // Merge with other bboxes
        bbox.Merge(xylim);
    }

    _addMarginToBoundingBox(bbox, margin, proj);

    // Return the estimated bounding box
    return bbox;
}

static bool _isValid(isce::geometry::BoundingBox bbox) {
    auto valid = [](double x) {
        return not (std::isnan(x) or std::isinf(x));
    };
    return valid(bbox.MinX) and valid(bbox.MaxX)
       and valid(bbox.MinY) and valid(bbox.MaxY);
}

static isce::geometry::BoundingBox _getGeoBoundingBoxBinarySearch(
        const isce::product::RadarGridParameters& radarGrid,
        const isce::core::Orbit& orbit, const isce::core::ProjectionBase* proj,
        const isce::core::LUT2d<double>& doppler, double min_height,
        double max_height, const double margin, const int pointsPerEdge,
        const double threshold, const int numiter,
        bool find_lowest_valid_height, const double height_threshold) {

    // Check for number of points on edge
    if (margin < 0.) {
        std::string errstr = "Margin should be a positive number. " +
                             std::to_string(margin) + " requested. ";
        throw isce::except::OutOfRange(ISCE_SRCINFO(), errstr);
    }

    // Initialize data structure for final output
    double mid_height = (min_height + max_height) / 2.0;

    isce::geometry::BoundingBox bbox = isce::geometry::getGeoBoundingBox(
            radarGrid, orbit, proj, doppler, {mid_height}, margin,
            pointsPerEdge, threshold, numiter, true);

    if (mid_height - min_height < height_threshold) {
        return bbox;
    }

    double new_min_height, new_max_height;
    // ^ is the XOR operator
    if (_isValid(bbox) ^ find_lowest_valid_height) {
        // higher height search
        new_min_height = mid_height;
        new_max_height = max_height;
    } else {
        // lower height search
        new_min_height = min_height;
        new_max_height = mid_height;
    }
    isce::geometry::BoundingBox bbox_result = _getGeoBoundingBoxBinarySearch(
            radarGrid, orbit, proj, doppler, new_min_height, new_max_height,
            margin, pointsPerEdge, threshold, numiter, find_lowest_valid_height,
            height_threshold);
    return bbox_result;
}

isce::geometry::BoundingBox isce::geometry::getGeoBoundingBoxHeightSearch(
        const isce::product::RadarGridParameters& radarGrid,
        const isce::core::Orbit& orbit, const isce::core::ProjectionBase* proj,
        const isce::core::LUT2d<double>& doppler, double min_height,
        double max_height, const double margin, const int pointsPerEdge,
        const double threshold, const int numiter,
        const double height_threshold) {

    // Check for number of points on edge
    if (margin < 0.) {
        std::string errstr = "Margin should be a positive number. " +
                             std::to_string(margin) + " requested. ";
        throw isce::except::OutOfRange(ISCE_SRCINFO(), errstr);
    }

    // Initialize data structure for final output
    const double margin_zero = 0;

    // Get perimeter for constant height
    BoundingBox bbox_min = getGeoBoundingBox(
            radarGrid, orbit, proj, doppler, {min_height}, margin_zero,
            pointsPerEdge, threshold, numiter, true);

    BoundingBox bbox_max = getGeoBoundingBox(
            radarGrid, orbit, proj, doppler, {max_height}, margin_zero,
            pointsPerEdge, threshold, numiter, true);

    if (!_isValid(bbox_min) && !_isValid(bbox_max)) {
        // both are invalid, skipping...
        return bbox_min;
    }

    if (_isValid(bbox_min) && !_isValid(bbox_max)) {
        // only lower height is valid
        bool find_lowest_valid_height = false;
        BoundingBox bbox = _getGeoBoundingBoxBinarySearch(
                radarGrid, orbit, proj, doppler, min_height, max_height,
                margin_zero, pointsPerEdge, threshold, numiter,
                find_lowest_valid_height, height_threshold);
        return bbox;
    }

    if (!_isValid(bbox_min) && _isValid(bbox_max)) {
        // only upper height is valid

        Vec3 sat_pos_mid, vel_mid, satLLH;
        double az_time_mid = radarGrid.sensingMid();
        orbit.interpolate(&sat_pos_mid, &vel_mid, az_time_mid,
                          isce::core::OrbitInterpBorderMode::FillNaN);

        const isce::core::Ellipsoid& ellipsoid = proj->ellipsoid();
        ellipsoid.xyzToLonLat(sat_pos_mid, satLLH);
        const double new_height =
                satLLH[2] - radarGrid.startingRange() + height_threshold * 0.5;
        if (new_height > min_height) {
            bbox_min = getGeoBoundingBox(
                    radarGrid, orbit, proj, doppler, {new_height}, margin_zero,
                    pointsPerEdge, threshold, numiter, true);
        }

        if (!_isValid(bbox_min)) {
            bool find_lowest_valid_height = true;
            BoundingBox bbox = _getGeoBoundingBoxBinarySearch(
                    radarGrid, orbit, proj, doppler, new_height, max_height,
                    margin_zero, pointsPerEdge, threshold, numiter,
                    find_lowest_valid_height, height_threshold);
            return bbox;
        }
    }

    // Both limits are valid
    bbox_min.Merge(bbox_max);

    _addMarginToBoundingBox(bbox_min, margin, proj);

    return bbox_min;
}
//end of file
