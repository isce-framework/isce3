#include "loadDem.h"
#include <isce3/except/Error.h>

using isce3::core::Vec3;

namespace isce3 { namespace geometry {


DEMInterpolator DEMRasterToInterpolator(
        isce3::io::Raster& demRaster,
        const isce3::product::GeoGridParameters& geoGrid,
        const int demMarginInPixels,
        const isce3::core::dataInterpMethod demInterpMethod)
{
        int lineStart = 0;
        int blockLength = geoGrid.length();
        int blockWidth = geoGrid.width();
        return DEMRasterToInterpolator(demRaster, geoGrid, lineStart,
                blockLength, blockWidth, demMarginInPixels, demInterpMethod);
}

DEMInterpolator DEMRasterToInterpolator(
        isce3::io::Raster& demRaster,
        const isce3::product::GeoGridParameters& geoGrid, const int lineStart,
        const int blockLength, const int blockWidth,
        const int demMarginInPixels,
        const isce3::core::dataInterpMethod demInterpMethod)
{
    // Get the debug journal
    pyre::journal::debug_t debug("isce.geometry.loadDem.DEMRasterToInterpolator");

    // DEM interpolator
    DEMInterpolator demInterp(0, demInterpMethod);

    // the epsg code of the input DEM
    int epsgcode = demRaster.getEPSG();

    // Initialize bounds
    double minX = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::lowest();

    // If the projection systems are different
    if (epsgcode != geoGrid.epsg()) {
        std::unique_ptr<isce3::core::ProjectionBase> proj(
                isce3::core::createProj(geoGrid.epsg()));

        // Create transformer to match the DEM
        std::unique_ptr<isce3::core::ProjectionBase> demproj(
                isce3::core::createProj(epsgcode));

        // Skip factors
        const int askip = std::max(static_cast<int>(blockLength / 10.), 1);
        const int rskip = std::max(static_cast<int>(blockWidth / 10.), 1);

        // Construct vectors of line/pixel indices to traverse perimeter
        std::vector<int> lineInd, pixInd;

        // Top edge
        for (int j = 0; j < blockWidth; j += rskip) {
            lineInd.push_back(0);
            pixInd.push_back(j);
        }

        // Right edge
        for (int i = 0; i < blockLength; i += askip) {
            lineInd.push_back(i);
            pixInd.push_back(blockWidth);
        }

        // Bottom edge
        for (int j = blockWidth; j > 0; j -= rskip) {
            lineInd.push_back(blockLength - 1);
            pixInd.push_back(j);
        }

        // Left edge
        for (int i = blockLength; i > 0; i -= askip) {
            lineInd.push_back(i);
            pixInd.push_back(0);
        }

        // Loop over the indices
        for (size_t i = 0; i < lineInd.size(); i++) {
            isce3::core::Vec3 outpt = {
                    geoGrid.startX() + geoGrid.spacingX() * pixInd[i],
                    geoGrid.startY() + geoGrid.spacingY() * (lineStart + lineInd[i]), 0.0};

            isce3::core::Vec3 dempt;
            if (!projTransform(proj.get(), demproj.get(), outpt, dempt)) {
                minX = std::min(minX, dempt[0]);
                maxX = std::max(maxX, dempt[0]);
                minY = std::min(minY, dempt[1]);
                maxY = std::max(maxY, dempt[1]);
            } else {
                std::string errmsg = "projection transformation between geogrid and DEM failed";
                throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errmsg);

            }
        }
    } else {
        // Use the corners directly as the projection system is the same
        maxY = geoGrid.startY() + geoGrid.spacingY() * lineStart;
        minY = geoGrid.startY() +
               geoGrid.spacingY() * (lineStart + blockLength - 1);
        minX = geoGrid.startX();
        maxX = geoGrid.startX() + geoGrid.spacingX() * (blockWidth - 1);
    }

    // Account for margins
    minX -= demMarginInPixels * demRaster.dx();
    maxX += demMarginInPixels * demRaster.dx();
    minY -= demMarginInPixels * std::abs(demRaster.dy());
    maxY += demMarginInPixels * std::abs(demRaster.dy());

    debug << minX << " , " << maxX << " , " << minY << ", " << maxY
              << pyre::journal::endl;

    // load the DEM for this bounding box
    demInterp.loadDEM(demRaster, minX, maxX, minY, maxY);
    debug << "DEM interpolation Done" << pyre::journal::endl;

    if (demInterp.width() == 0 || demInterp.length() == 0) {
        pyre::journal::warning_t warning("isce.geometry.loadDem.DEMRasterToInterpolator");
        warning << "there is not enough DEM coverage in the bounding box. "
                << pyre::journal::endl;
    }
    // declare the dem interpolator
    demInterp.declare();

    return demInterp;
}

isce3::error::ErrorCode loadDemFromProj(
    isce3::io::Raster& dem_raster, const double x0, const double xf,
    const double y0, const double yf,
    DEMInterpolator* dem_interp,
    isce3::core::ProjectionBase* proj, const int dem_margin_x_in_pixels,
    const int dem_margin_y_in_pixels, const int dem_raster_band,
    const int n_edge_samples){
    double min_x, max_x, min_y, max_y;

    min_y = std::min(y0, yf);
    max_y = std::max(y0, yf);

    min_x = std::min(x0, xf);
    max_x = std::max(x0, xf);

    // Test for antimeridian crossing (EPSG 4326) and unwrap X (longitude)
    // coordinates if `max_x - min_x` is greater than 180 degrees
    if ((((proj == nullptr) && (dem_raster.getEPSG() == 4326)) ||
         ((proj != nullptr) && (proj->code() == 4326))) &&
        (max_x - min_x > 180)) {
            const double x0_unwrapped = x0 < 0 ? x0 + 360.0 : x0;
            const double xf_unwrapped = xf < 0 ? xf + 360.0 : xf;
            min_x = std::min(x0_unwrapped, xf_unwrapped);
            max_x = std::max(x0_unwrapped, xf_unwrapped);
    }

    if (proj != nullptr && proj->code() != dem_raster.getEPSG()) {

        std::unique_ptr<isce3::core::ProjectionBase> dem_proj(
                isce3::core::createProj(dem_raster.getEPSG()));

        const int N = std::max(n_edge_samples, 2);

        // Densely sample all four edges of the input bounding box
        // to capture curvature introduced by reprojection
        // (e.g. UTM -> geographic).
        std::vector<double> all_x, all_y;
        all_x.reserve(4 * N);
        all_y.reserve(4 * N);

        for (int i = 0; i < N; ++i) {
            double t = static_cast<double>(i) / (N - 1);
            double x_mid = min_x + t * (max_x - min_x);
            double y_mid = min_y + t * (max_y - min_y);

            // Left edge (x = min_x, y varies)
            auto left_llh = proj->inverse({min_x, y_mid, 0});
            auto left_xy  = dem_proj->forward(left_llh);
            all_x.push_back(left_xy[0]);
            all_y.push_back(left_xy[1]);

            // Right edge (x = max_x, y varies)
            auto right_llh = proj->inverse({max_x, y_mid, 0});
            auto right_xy  = dem_proj->forward(right_llh);
            all_x.push_back(right_xy[0]);
            all_y.push_back(right_xy[1]);

            // Bottom edge (y = min_y, x varies)
            auto bottom_llh = proj->inverse({x_mid, min_y, 0});
            auto bottom_xy  = dem_proj->forward(bottom_llh);
            all_x.push_back(bottom_xy[0]);
            all_y.push_back(bottom_xy[1]);

            // Top edge (y = max_y, x varies)
            auto top_llh = proj->inverse({x_mid, max_y, 0});
            auto top_xy  = dem_proj->forward(top_llh);
            all_x.push_back(top_xy[0]);
            all_y.push_back(top_xy[1]);
        }

        min_y = *std::min_element(all_y.begin(), all_y.end());
        max_y = *std::max_element(all_y.begin(), all_y.end());

        min_x = *std::min_element(all_x.begin(), all_x.end());
        max_x = *std::max_element(all_x.begin(), all_x.end());

        // If the DEM is in geographic coordinates and the X range
        // exceeds 180 degrees, an antimeridian crossing is likely.
        // Retry in [0, 360] domain.
        if (dem_raster.getEPSG() == 4326 && (max_x - min_x) > 180.0) {
            min_x = std::numeric_limits<double>::max();
            max_x = std::numeric_limits<double>::lowest();
            for (double lon : all_x) {
                double lon_360 = lon < 0 ? lon + 360.0 : lon;
                min_x = std::min(min_x, lon_360);
                max_x = std::max(max_x, lon_360);
            }
        }
    }

    double margin_y = dem_margin_y_in_pixels * std::abs(dem_raster.dy());
    min_y -= margin_y;
    max_y += margin_y;

    double margin_x = dem_margin_x_in_pixels * std::abs(dem_raster.dx());
    min_x -= margin_x;
    max_x += margin_x;

    // If DEM coordinates are in geographic, ensure latitude values
    // fall between [-90.0, 90.0] after applying `margin_y`
    if (dem_raster.getEPSG() == 4326) {
        min_y = std::clamp(min_y, -90.0, 90.0);
        max_y = std::clamp(max_y, -90.0, 90.0);
    }

    isce3::error::ErrorCode error_code;
    _Pragma("omp critical")
    {
        error_code = dem_interp->loadDEM(
                dem_raster, min_x, max_x, min_y, max_y, dem_raster_band);
    }
    return error_code;
}

Vec3 getDemCoordsSameEpsg(double x, double y,
        const DEMInterpolator& dem_interp, isce3::core::ProjectionBase*)
{

    Vec3 dem_coords = {x, y, dem_interp.interpolateXY(x, y)};
    return dem_coords;
}

Vec3 getDemCoordsDiffEpsg(double x, double y,
        const DEMInterpolator& dem_interp,
        isce3::core::ProjectionBase* input_proj)
{

    auto input_coords_llh = input_proj->inverse({x, y, 0});
    Vec3 dem_vect;
    dem_interp.proj()->forward(input_coords_llh, dem_vect);
    Vec3 dem_coords = {dem_vect[0], dem_vect[1],
            dem_interp.interpolateXY(dem_vect[0], dem_vect[1])};

    return dem_coords;
}

}}
