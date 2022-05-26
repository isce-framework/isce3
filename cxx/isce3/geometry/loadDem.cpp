#include "loadDem.h"
#include <isce3/except/Error.h>

using isce3::core::Vec3;

namespace isce3 { namespace geometry {

DEMInterpolator loadDEM(
        isce3::io::Raster& demRaster,
        const isce3::product::GeoGridParameters& geoGrid, int lineStart,
        int blockLength, int blockWidth, const int demMarginInPixels,
        isce3::core::dataInterpMethod demInterpMethod)
{
    // DEM interpolator
    DEMInterpolator demInterp(0, demInterpMethod);

    // the epsg code of the input DEM
    int epsgcode = demRaster.getEPSG();

    // Initialize bounds
    double minX = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::min();
    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::min();

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

    std::cout << minX << " , " << maxX << " , " << minY << ", " << maxY
              << std::endl;

    // load the DEM for this bounding box
    demInterp.loadDEM(demRaster, minX, maxX, minY, maxY);
    std::cout << "DEM interpolation Done" << std::endl;

    if (demInterp.width() == 0 || demInterp.length() == 0)
        std::cout << "warning there are not enough DEM coverage in the "
                     "bounding box. "
                  << std::endl;

    // declare the dem interpolator
    demInterp.declare();

    return demInterp;
}


isce3::error::ErrorCode loadDemFromProj(
    isce3::io::Raster& dem_raster, const double minX, const double maxX,
    const double minY, const double maxY, 
    DEMInterpolator* dem_interp,
    isce3::core::ProjectionBase* proj, const int dem_margin_x_in_pixels,
    const int dem_margin_y_in_pixels, const int dem_raster_band) {

    Vec3 geogrid_min_xy = {minX, std::min(minY, maxY), 0};
    Vec3 geogrid_max_xy = {maxX, std::max(minY, maxY), 0};
    double min_x, max_x, min_y, max_y;

    if (proj == nullptr || proj->code() == dem_raster.getEPSG()) {

        Vec3 dem_min_xy, dem_max_xy;

        dem_min_xy = geogrid_min_xy;
        dem_max_xy = geogrid_max_xy;
        min_x = dem_min_xy[0];
        max_x = dem_max_xy[0];
        min_y = dem_min_xy[1];
        max_y = dem_max_xy[1];
    } else {
        std::unique_ptr<isce3::core::ProjectionBase> dem_proj(
                isce3::core::createProj(dem_raster.getEPSG()));
        auto p1_llh = proj->inverse({geogrid_min_xy[0], geogrid_min_xy[1], 0});
        auto p2_llh = proj->inverse({geogrid_min_xy[0], geogrid_max_xy[1], 0});
        auto p3_llh = proj->inverse({geogrid_max_xy[0], geogrid_min_xy[1], 0});
        auto p4_llh = proj->inverse({geogrid_max_xy[0], geogrid_max_xy[1], 0});

        Vec3 p1_xy, p2_xy, p3_xy, p4_xy;

        dem_proj->forward(p1_llh, p1_xy);
        dem_proj->forward(p2_llh, p2_xy);
        dem_proj->forward(p3_llh, p3_xy);
        dem_proj->forward(p4_llh, p4_xy);
        min_x = std::min(
                std::min(p1_xy[0], p2_xy[0]), std::min(p3_xy[0], p4_xy[0]));
        max_x = std::max(
                std::max(p1_xy[0], p2_xy[0]), std::max(p3_xy[0], p4_xy[0]));
        min_y = std::min(
                std::min(p1_xy[1], p2_xy[1]), std::min(p3_xy[1], p4_xy[1]));
        max_y = std::max(
                std::max(p1_xy[1], p2_xy[1]), std::max(p3_xy[1], p4_xy[1]));
    }

    float margin_x = dem_margin_x_in_pixels * dem_raster.dx();
    float margin_y = dem_margin_y_in_pixels * std::abs(dem_raster.dy());

    min_x -= margin_x;
    max_x += margin_x;
    min_y -= margin_y;
    max_y += margin_y;

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
