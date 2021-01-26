// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Piyush Agram
// Copyright 2019

// header file
#include "boundingbox.h"

// cassert for assert()
#include <cassert>

// pyre::journal
#include <pyre/journal.h>

// isce3::core
#include <isce3/core/Basis.h>
#include <isce3/core/Constants.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Pixel.h>
#include <isce3/core/Projections.h>

// isce3::geometry
#include <isce3/geometry/geometry.h>

// isce3::except
#include <isce3/except/Error.h>

// isce3::geometry
#include "DEMInterpolator.h"

// pull in some isce3::core namespaces
using isce3::core::Vec3;
using isce3::core::ProjectionBase;
using isce3::core::Basis;

isce3::geometry::Perimeter
isce3::geometry::
getGeoPerimeter(const isce3::product::RadarGridParameters &radarGrid,
                const isce3::core::Orbit &orbit,
                const isce3::core::ProjectionBase *proj,
                const isce3::core::LUT2d<double> &doppler,
                const isce3::geometry::DEMInterpolator &demInterp,
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
        throw isce3::except::OutOfRange(ISCE_SRCINFO(), errstr); 
    }

    //Journal for warning
    pyre::journal::warning_t warning("isce.geometry.perimeter");

    //Initialize results
    isce3::geometry::Perimeter perimeter;

    //Ellipsoid being used for processing
    const isce3::core::Ellipsoid &ellipsoid = proj->ellipsoid();

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
            throw isce3::except::OutOfRange(ISCE_SRCINFO(), errstr);
        }

        //Add transformed point to the perimeter
        perimeter.addPoint(mapxyz[0], mapxyz[1], mapxyz[2]);

    }

    //Ensure polygon is closed
    perimeter.closeRings();

    //Return points
    return perimeter;
}

static void _addMarginToBoundingBox(isce3::geometry::BoundingBox& bbox,
                                    const double margin,
                                    const isce3::core::ProjectionBase* proj) {

    // Set up margin in meters / degrees
    double delta = margin;
    if (proj->code() != 4326)
        delta = isce3::core::decimaldeg2meters(margin);

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

isce3::geometry::BoundingBox isce3::geometry::getGeoBoundingBox(
        const isce3::product::RadarGridParameters& radarGrid,
        const isce3::core::Orbit& orbit, const isce3::core::ProjectionBase* proj,
        const isce3::core::LUT2d<double>& doppler,
        const std::vector<double>& hgts, const double margin,
        const int pointsPerEdge, const double threshold, const int numiter,
        bool ignore_out_of_range_exception) {

    // Check for number of points on edge
    if (margin < 0.) {
        std::string errstr = "Margin should be a positive number. " +
                             std::to_string(margin) + " requested. ";
        throw isce3::except::OutOfRange(ISCE_SRCINFO(), errstr);
    }

    // Initialize data structure for final output
    isce3::geometry::BoundingBox bbox;

    // Loop over the heights
    for (const auto& height : hgts) {
        // Get perimeter for constant height
        isce3::geometry::DEMInterpolator constDEM(height);
        isce3::geometry::Perimeter perimeter;

        if (ignore_out_of_range_exception) {
            try {
                perimeter = getGeoPerimeter(radarGrid, orbit, proj, doppler,
                                            constDEM, pointsPerEdge, threshold,
                                            numiter);
            } catch (const isce3::except::OutOfRange&) {
                continue;
            }
        } else {
            perimeter =
                    getGeoPerimeter(radarGrid, orbit, proj, doppler, constDEM,
                                    pointsPerEdge, threshold, numiter);
        }

        // Get bounding box for given height
        isce3::geometry::BoundingBox xylim;
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

static bool _isValid(isce3::geometry::BoundingBox bbox) {
    auto valid = [](double x) {
        return not (std::isnan(x) or std::isinf(x));
    };
    return valid(bbox.MinX) and valid(bbox.MaxX)
       and valid(bbox.MinY) and valid(bbox.MaxY);
}

static isce3::geometry::BoundingBox _getGeoBoundingBoxBinarySearch(
        const isce3::product::RadarGridParameters& radarGrid,
        const isce3::core::Orbit& orbit,
        const isce3::core::ProjectionBase* proj,
        const isce3::core::LUT2d<double>& doppler, double min_height,
        double max_height, const double margin, const int pointsPerEdge,
        const double threshold, const int numiter,
        bool find_lowest_valid_height,
        isce3::geometry::BoundingBox bbox_best_solution_from_other_end,
        const double height_threshold)
{

    // Check input arguments
    if (max_height < min_height) {
        std::string errstr = "max_height <  min_height";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errstr);
    }

    if (margin < 0.) {
        std::string errstr = "Margin should be a positive number. " +
                             std::to_string(margin) + " requested. ";
        throw isce3::except::OutOfRange(ISCE_SRCINFO(), errstr);
    }

    // Initialize data structure for final output
    double mid_height = (min_height + max_height) / 2.0;

    isce3::geometry::BoundingBox bbox_mid = isce3::geometry::getGeoBoundingBox(
            radarGrid, orbit, proj, doppler, {mid_height}, margin,
            pointsPerEdge, threshold, numiter, true);

    if (mid_height - min_height < height_threshold && _isValid(bbox_mid)) {
        return bbox_mid;
    } else if (mid_height - min_height < height_threshold) {
        return bbox_best_solution_from_other_end;
    }

    double new_min_height, new_max_height;
    // ^ is the XOR operator
    if (_isValid(bbox_mid) ^ find_lowest_valid_height) {
        // higher height search
        // (i.e. mid is   valid and looking for highest or
        //       mid is invalid and looking for lowest)
        new_min_height = mid_height;
        new_max_height = max_height;
    } else {
        // lower height search
        new_min_height = min_height;
        new_max_height = mid_height;
    }

    if (_isValid(bbox_mid))
        bbox_best_solution_from_other_end = bbox_mid;

    isce3::geometry::BoundingBox bbox_result = _getGeoBoundingBoxBinarySearch(
            radarGrid, orbit, proj, doppler, new_min_height, new_max_height,
            margin, pointsPerEdge, threshold, numiter, find_lowest_valid_height,
            bbox_best_solution_from_other_end, height_threshold);

    return bbox_result;
}

isce3::geometry::BoundingBox isce3::geometry::getGeoBoundingBoxHeightSearch(
        const isce3::product::RadarGridParameters& radarGrid,
        const isce3::core::Orbit& orbit, const isce3::core::ProjectionBase* proj,
        const isce3::core::LUT2d<double>& doppler, double min_height,
        double max_height, const double margin, const int pointsPerEdge,
        const double threshold, const int numiter,
        const double height_threshold) {

    // Check input arguments
    if (max_height < min_height) {
        std::string errstr = "max_height <  min_height";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errstr);
    }
    if (margin < 0.) {
        std::string errstr = "Margin should be a positive number. " +
                             std::to_string(margin) + " requested. ";
        throw isce3::except::OutOfRange(ISCE_SRCINFO(), errstr);
    }

    assert(max_height >= min_height);

    // Initialize data structure for final output
    const double margin_zero = 0;

    // Get BBox for min_height
    BoundingBox bbox_min = getGeoBoundingBox(
            radarGrid, orbit, proj, doppler, {min_height}, margin_zero,
            pointsPerEdge, threshold, numiter, true);

    if (max_height == min_height && !_isValid(bbox_min)) {
        std::string errstr = "Bounding box not found for given parameters.";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errstr);
    } 
    else if (max_height == min_height) {
        return bbox_min;
    }

    // Get BBox for max_height
    BoundingBox bbox_max = getGeoBoundingBox(
            radarGrid, orbit, proj, doppler, {max_height}, margin_zero,
            pointsPerEdge, threshold, numiter, true);

    if (!_isValid(bbox_min) && !_isValid(bbox_max)) {
        // both are invalid
        std::string errstr = "Bounding box not found for given parameters.";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errstr);
    }

    else if (_isValid(bbox_min) && !_isValid(bbox_max)) {
        // only lower height is valid
        bool find_lowest_valid_height = false;
        bbox_max = _getGeoBoundingBoxBinarySearch(
                radarGrid, orbit, proj, doppler, min_height, max_height,
                margin_zero, pointsPerEdge, threshold, numiter,
                find_lowest_valid_height, bbox_min, height_threshold);
    } else if (!_isValid(bbox_min) && _isValid(bbox_max)) {
        // only upper height is valid

        Vec3 sat_pos_mid, vel_mid, satLLH;
        double az_time_mid = radarGrid.sensingMid();
        orbit.interpolate(&sat_pos_mid, &vel_mid, az_time_mid,
                          isce3::core::OrbitInterpBorderMode::FillNaN);

        const isce3::core::Ellipsoid& ellipsoid = proj->ellipsoid();
        ellipsoid.xyzToLonLat(sat_pos_mid, satLLH);
        const double new_height =
                satLLH[2] - radarGrid.startingRange() + height_threshold * 0.5;

        if (new_height > min_height) {
            bbox_min = getGeoBoundingBox(
                    radarGrid, orbit, proj, doppler, {new_height}, margin_zero,
                    pointsPerEdge, threshold, numiter, true);
            min_height = new_height;
        }

        if (!_isValid(bbox_min)) {
            bool find_lowest_valid_height = true;
            bbox_min = _getGeoBoundingBoxBinarySearch(
                    radarGrid, orbit, proj, doppler, min_height, max_height,
                    margin_zero, pointsPerEdge, threshold, numiter,
                    find_lowest_valid_height, bbox_max, height_threshold);
        }
    }

    // Both limits are valid
    bbox_min.Merge(bbox_max);

    if (!_isValid(bbox_min)) {
        // if result is invalid
        std::string errstr = "Bounding box not found for given parameters.";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errstr);
    }

    _addMarginToBoundingBox(bbox_min, margin, proj);

    return bbox_min;
}
//end of file
