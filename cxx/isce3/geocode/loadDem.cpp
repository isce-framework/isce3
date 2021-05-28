#include "loadDem.h"
#include <isce3/except/Error.h>

isce3::geometry::DEMInterpolator
isce3::geocode::loadDEM(isce3::io::Raster& demRaster,
                       const isce3::product::GeoGridParameters& geoGrid,
                       int lineStart, int blockLength, int blockWidth,
                       double demMargin,
                       isce3::core::dataInterpMethod demInterpMethod)
{
    // DEM interpolator
    isce3::geometry::DEMInterpolator demInterp(0, demInterpMethod);

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

    // If not LonLat, scale to meters
    demMargin = (epsgcode != 4326) ? isce3::core::decimaldeg2meters(demMargin)
                                   : demMargin;

    // Account for margins
    minX -= demMargin;
    maxX += demMargin;
    minY -= demMargin;
    maxY += demMargin;

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
