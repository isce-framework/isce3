#include "loadDem.h"

isce::geometry::DEMInterpolator isce::geocode::loadDEM(
        isce::io::Raster demRaster,
        isce::core::ProjectionBase * proj,
        const isce::product::GeoGridParameters & geoGrid,
        int lineStart, int blockLength,
        int blockWidth, double demMargin)
{
    isce::geometry::DEMInterpolator demInterp;
    //Create projection for DEM
    int epsgcode = demRaster.getEPSG();

    //Initialize bounds
    double minX = std::numeric_limits<double>::min();
    double maxX = std::numeric_limits<double>::max();
    double minY = std::numeric_limits<double>::min();
    double maxY = std::numeric_limits<double>::max();

    //Projection systems are different
    if (epsgcode != geoGrid.epsg())
    {

        //Create transformer to match the DEM
        //isce::core::ProjectionBase *demproj = isce::core::createProj(epsgcode);
        std::unique_ptr<isce::core::ProjectionBase> demproj(
                isce::core::createProj(epsgcode));

        //Skip factors
        const int askip = std::max( static_cast<int>(blockLength / 10.), 1);
        const int rskip = std::max( static_cast<int>(blockWidth / 10.), 1);


        //Construct vectors of line/pixel indices to traverse perimeter
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

        //Loop over the indices
        for (size_t i = 0; i < lineInd.size(); i++)
        {
            isce::core::Vec3 outpt = { geoGrid.startX() + geoGrid.spacingX() * pixInd[i],
                           geoGrid.startY() + geoGrid.spacingY() * lineInd[i],
                           0.0};

            isce::core::Vec3 dempt;
            if (!projTransform(proj, demproj.get(), outpt, dempt))
            {
                minX = std::min(minX, dempt[0]);
                maxX = std::max(maxX, dempt[0]);
                minY = std::min(minY, dempt[1]);
                maxY = std::max(maxY, dempt[1]);
            }
        }
    }
    else
    {
        // Use the corners directly as the projection system is the same
        maxY = geoGrid.startY() + geoGrid.spacingY()*lineStart;
        minY = geoGrid.startY() + geoGrid.spacingY()*(lineStart + blockLength - 1);
        minX = geoGrid.startX();
        maxX = geoGrid.startX() + geoGrid.spacingX()*(blockWidth - 1);

    }

    //If not LonLat, scale to meters
    demMargin = (epsgcode != 4326)? isce::core::decimaldeg2meters(demMargin) : demMargin;

    // Account for margins
    minX -= demMargin;
    maxX += demMargin;
    minY -= demMargin;
    maxY += demMargin;

    std::cout << minX << " , " << maxX << " , " << minY << ", " << maxY << std::endl;

    // load the DEM for this bounding box
    demInterp.loadDEM(demRaster, minX, maxX, minY, maxY);
    std::cout << "DEM interpolation Done" << std::endl;

    if (demInterp.width() == 0 || demInterp.length() == 0)
        std::cout << "warning there are not enough DEM coverage in the bounding box. " << std::endl;

    // declare the dem interpolator
    demInterp.declare();

    return demInterp;
}
