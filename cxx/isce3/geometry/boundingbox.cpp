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
#include <isce3/geometry/rdr2geo_roots.h>

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
                const double threshold)
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
    // Exclude final point.  Will force first and last point to be the same.
    for (int ii = pointsPerEdge-2; ii > 0; ii--)
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
        // Careful to initialize height since it's the initial guess.
        Vec3 xyz, llh, mapxyz;

        //Run rdr2geo
        auto converged = rdr2geo_bracket(tline, rng, dopp, orbit, demInterp,
            xyz, radarGrid.wavelength(), radarGrid.lookSide(), threshold);
        
        ellipsoid.xyzToLonLat(xyz, llh);

        //Transform point to projection
        int status = proj->forward(llh, mapxyz);
        if (status or not converged)
        {
            std::string errstr = "Error in transforming point (" + std::to_string(llh[0]) +
                                 "," + std::to_string(llh[1]) + "," + std::to_string(llh[2]) +
                                 ") to projection EPSG:" + std::to_string(proj->code());
            throw isce3::except::OutOfRange(ISCE_SRCINFO(), errstr);
        }

        //Add transformed point to the perimeter
        perimeter.addPoint(mapxyz[0], mapxyz[1], mapxyz[2]);

    }

    // Ensure polygon is closed.  Since we skipped the last point, this will
    // always add one final point that is exactly equal to the first.
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
    bbox.MaxY += delta;

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
        const int pointsPerEdge, const double threshold,
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
                                            constDEM, pointsPerEdge, threshold);
                                            
            } catch (const isce3::except::OutOfRange&) {
                continue;
            }
        } else {
            perimeter =
                    getGeoPerimeter(radarGrid, orbit, proj, doppler, constDEM,
                                    pointsPerEdge, threshold);
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
        const double threshold,
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
            pointsPerEdge, threshold, true);

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
            margin, pointsPerEdge, threshold, find_lowest_valid_height,
            bbox_best_solution_from_other_end, height_threshold);

    return bbox_result;
}

isce3::geometry::BoundingBox isce3::geometry::getGeoBoundingBoxHeightSearch(
        const isce3::product::RadarGridParameters& radarGrid,
        const isce3::core::Orbit& orbit, const isce3::core::ProjectionBase* proj,
        const isce3::core::LUT2d<double>& doppler, double min_height,
        double max_height, const double margin, const int pointsPerEdge,
        const double threshold,
        const double height_threshold) {

    // Check input arguments
    if (max_height < min_height) {
        std::string errstr = "max_height <  min_height";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errstr);
    }
    if (margin < 0.) {
        std::string errstr = "Margin should be a positive number. " +
                             std::to_string(margin) + " requested.";
        throw isce3::except::OutOfRange(ISCE_SRCINFO(), errstr);
    }

    assert(max_height >= min_height);

    // Initialize data structure for final output
    const double margin_zero = 0;

    // Get BBox for min_height
    BoundingBox bbox_min = getGeoBoundingBox(
            radarGrid, orbit, proj, doppler, {min_height}, margin_zero,
            pointsPerEdge, threshold, true);

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
            pointsPerEdge, threshold, true);

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
                margin_zero, pointsPerEdge, threshold,
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
                    pointsPerEdge, threshold, true);
            min_height = new_height;
        }

        if (!_isValid(bbox_min)) {
            bool find_lowest_valid_height = true;
            bbox_min = _getGeoBoundingBoxBinarySearch(
                    radarGrid, orbit, proj, doppler, min_height, max_height,
                    margin_zero, pointsPerEdge, threshold,
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

isce3::geometry::RadarGridBoundingBox isce3::geometry::getRadarBoundingBox(
        const isce3::product::GeoGridParameters& geo_grid,
        const isce3::product::RadarGridParameters& radar_grid,
        const isce3::core::Orbit& orbit,
        isce3::geometry::DEMInterpolator& dem_interp,
        const isce3::core::LUT2d<double>& doppler,
        const int interp_margin,
        const isce3::geometry::detail::Geo2RdrParams& g2r_params,
        const int geogrid_expansion_threshold)
{
    // Compute extreme heights of DEM raster for geo2rdr computations
    // min and max heights to be used with x and y values built from geo_grid
    float min_height, max_height, mean_height;
    dem_interp.computeMinMaxMeanHeight(min_height, max_height, mean_height);

    const auto rdr_bbox = isce3::geometry::getRadarBoundingBox(
            geo_grid, radar_grid, orbit, min_height, max_height, doppler,
            interp_margin, g2r_params, geogrid_expansion_threshold);
    return rdr_bbox;
}

isce3::geometry::RadarGridBoundingBox isce3::geometry::getRadarBoundingBox(
        const isce3::product::GeoGridParameters& geo_grid,
        const isce3::product::RadarGridParameters& radar_grid,
        const isce3::core::Orbit& orbit,
        const float min_height, const float max_height,
        const isce3::core::LUT2d<double>& doppler,
        const int interp_margin,
        const isce3::geometry::detail::Geo2RdrParams& g2r_params,
        const int geogrid_expansion_threshold)
{
    // Check input arguments
    if (max_height < min_height) {
        std::string errstr = "max_height <  min_height";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errstr);
    }

    if (interp_margin < 0) {
        std::string errstr = "Margin should be a nonnegative number. " +
                             std::to_string(interp_margin) + " requested.";
        throw isce3::except::OutOfRange(ISCE_SRCINFO(), errstr);
    }

    // min and max heights to be used with x and y values built from geo_grid
    // TODO sanity check values for min and max height - what are sane values?
    // between Marianas Trench and Everest?
    float height_range[2] = {min_height, max_height};

    isce3::geometry::RadarGridBoundingBox rdrBBox;

    // First and last line of the data block in radar coordinates
    rdrBBox.firstAzimuthLine = static_cast<int>(radar_grid.length() - 1);
    rdrBBox.lastAzimuthLine = 0;

    // First and last pixel of the data block in radar coordinates
    rdrBBox.firstRangeSample = static_cast<int>(radar_grid.width() - 1);
    rdrBBox.lastRangeSample = 0;

    auto proj = isce3::core::makeProjection(geo_grid.epsg());
    const isce3::core::Ellipsoid& ellipsoid = proj->ellipsoid();

    // two matrices whose indices indicates the expansion directions of
    // the geocoded grid in x and y directions.
    // If computing the radar grid for a corner of the geocoded grid
    // fails, we expand the geocoded boundary towards the outside of the
    // geo-grid and try again. This can be iterated until finding a converged
    // point or until exhausting the number of iterations of the expansion.
    isce3::core::Matrix<int> geogrid_expansion_x(2, 2);
    isce3::core::Matrix<int> geogrid_expansion_y(2, 2);

    // top-left corner
    geogrid_expansion_x(0,0) = -1;
    geogrid_expansion_y(0,0) = -1;

    // top-right corner
    geogrid_expansion_x(0,1) = 1;
    geogrid_expansion_y(0,1) = -1;

    // bottom-left corner
    geogrid_expansion_x(1,0) = -1;
    geogrid_expansion_y(1,0) = 1;

    // bottom-right corner
    geogrid_expansion_x(1,1) = 1;
    geogrid_expansion_y(1,1) = 1;

    // flag indicating if geo2rdr computation converges for all four
    // corners of the geocoded grid
    bool all_converged = true;

    std::string bad_corners = "";
    // looping over the four corners of the geocoded grid
    for (size_t line = 0; line < 2; ++line) {

        // y in the geocoded grid for current corner
        double y0 = geo_grid.startY() + geo_grid.spacingY() * line * geo_grid.length();

        for (size_t pixel = 0; pixel < 2; ++pixel) {

            // x in the geocoded grid for current corner
            double x0 = geo_grid.startX() + geo_grid.spacingX() * pixel * geo_grid.width();

            // check if geo2rdr converges for current corner
            bool corner_converge = false;

            // looping over range of heights (min and max height of the input DEM)
            for (size_t i_h_min_max = 0; i_h_min_max < 2; ++i_h_min_max) {
                // try with min and max height
                float height = height_range[i_h_min_max];

                double aztime = radar_grid.sensingMid();
                double srange = radar_grid.startingRange();

                // track number of time geo2rdr called for current corner
                // at min or max height
                int corner_iterations = 0;

                // init 0 = geo2rdr not converged
                int geostat = 0;
                while (geostat == 0
                        and corner_iterations < geogrid_expansion_threshold) {
                    // expand the geogrid corner. For the first iteration,
                    // there is no expansion. If geo2rdr converges, there will
                    // be no expansion.
                    double x = x0 + corner_iterations * geogrid_expansion_x(line, pixel) * geo_grid.spacingX();
                    double y = y0 + corner_iterations * geogrid_expansion_y(line, pixel) * geo_grid.spacingY();

                    // coordinate in the output projection system
                    const isce3::core::Vec3 xyz {x, y, 0.0};

                    // transform the xyz in the output projection system to llh
                    isce3::core::Vec3 llh = proj->inverse(xyz);

                    // interpolate the height from the DEM for this pixel
                    llh[2] = height;

                    geostat = isce3::geometry::geo2rdr(
                        llh, ellipsoid, orbit, doppler, aztime, srange,
                        radar_grid.wavelength(), radar_grid.lookSide(),
                        g2r_params.threshold, g2r_params.maxiter,
                        g2r_params.delta_range);

                    corner_iterations += 1;
                }

                // if no convergence over entire expansion threshold skip:
                // az/rg first/last boundary check
                // marking current corner as converged
                if (geostat == 0)
                    continue;

                // mark current corner as converged
                corner_converge = true;

                // get the row and column index in the radar grid
                double azimuth_coord = (aztime - radar_grid.sensingStart()) * radar_grid.prf();
                double range_coord = (srange - radar_grid.startingRange()) /
                              radar_grid.rangePixelSpacing();
                rdrBBox.firstAzimuthLine = std::min(
                        rdrBBox.firstAzimuthLine, static_cast<int>(std::floor(azimuth_coord)));
                rdrBBox.lastAzimuthLine = std::max(
                        rdrBBox.lastAzimuthLine, static_cast<int>(std::ceil(azimuth_coord)));
                rdrBBox.firstRangeSample = std::min(
                        rdrBBox.firstRangeSample, static_cast<int>(std::floor(range_coord)));
                rdrBBox.lastRangeSample = std::max(
                        rdrBBox.lastRangeSample, static_cast<int>(std::ceil(range_coord)));
            } // min and max height loop

            // if no corner converge, store failed corner and skip
            // first/last az/rg index computations
            if (!corner_converge) {
                std::string line_dir = geogrid_expansion_y(line, pixel) == 1 ? "north" : "south";
                std::string pixel_dir = geogrid_expansion_x(line, pixel) == 1 ? "east" : "west";
                bad_corners += "("  + line_dir + pixel_dir + ") ";
                all_converged = false;
                continue;
            }
        } // pixel loop
    } // line loop

    // check if the computed bounding box overlaps with
    // the input radar grid
    bool is_valid_bound = true;
    int max_azimuth = radar_grid.length() - 1;
    int max_range = radar_grid.width() - 1;

    if (!all_converged) {
        const std::string err_str =
        "ERROR: geo2rdr computations did not converge for at corner(s) "
        + bad_corners;

        throw isce3::except::RuntimeError(ISCE_SRCINFO(), err_str);
    }

    // Non-overlap warning to be appended with non-overlap specifics
    std::string err_str = "ERROR: The computed radar bounding box does not overlap with the input radar grid.";

    if ((rdrBBox.firstAzimuthLine <= 0) and (rdrBBox.lastAzimuthLine <= 0)) {
        err_str += " Azimuth first and last <= 0.";
        is_valid_bound = false;
    }
    if ((rdrBBox.firstAzimuthLine >= max_azimuth) and (rdrBBox.lastAzimuthLine >= max_azimuth)) {
        err_str += " Azimuth first and last >= max azimuth.";
        is_valid_bound = false;
    }
    if ((rdrBBox.firstRangeSample <= 0) and (rdrBBox.lastRangeSample <= 0)) {
        err_str += " Range first and last <= 0.";
        is_valid_bound = false;
    }
    if ((rdrBBox.firstRangeSample >= max_range) and (rdrBBox.lastRangeSample >= max_range)) {
        err_str += " Range first and last >= max range.";
        is_valid_bound = false;
    }

    if (!is_valid_bound) {
        throw isce3::except::RuntimeError(ISCE_SRCINFO(), err_str);
    }

    // get the bounding in radar coordinates with a margin
    rdrBBox.firstAzimuthLine = std::max(rdrBBox.firstAzimuthLine - interp_margin, 0);
    rdrBBox.firstRangeSample = std::max(rdrBBox.firstRangeSample - interp_margin, 0);

    rdrBBox.lastAzimuthLine = std::min(rdrBBox.lastAzimuthLine + interp_margin,
                               static_cast<int>(radar_grid.length() - 1));
    rdrBBox.lastRangeSample = std::min(rdrBBox.lastRangeSample + interp_margin,
                              static_cast<int>(radar_grid.width() - 1));

    return rdrBBox;
}
