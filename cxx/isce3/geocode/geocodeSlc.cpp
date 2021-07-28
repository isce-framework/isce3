#include "geocodeSlc.h"

#include <memory>

#include <isce3/core/Constants.h>
#include <isce3/core/Ellipsoid.h>
#include <isce3/core/LUT2d.h>
#include <isce3/core/Orbit.h>
#include <isce3/core/Projections.h>
#include <isce3/geocode/baseband.h>
#include <isce3/geocode/interpolate.h>
#include <isce3/geocode/loadDem.h>
#include <isce3/geometry/DEMInterpolator.h>
#include <isce3/geometry/geometry.h>
#include <isce3/io/Raster.h>
#include <isce3/product/GeoGridParameters.h>
#include <isce3/product/Product.h>
#include <isce3/product/RadarGridParameters.h>

void isce3::geocode::geocodeSlc(
        isce3::io::Raster& outputRaster, isce3::io::Raster& inputRaster,
        isce3::io::Raster& demRaster,
        const isce3::product::RadarGridParameters& radarGrid,
        const isce3::product::GeoGridParameters& geoGrid,
        const isce3::core::Orbit& orbit,
        const isce3::core::LUT2d<double>& nativeDoppler,
        const isce3::core::LUT2d<double>& imageGridDoppler,
        const isce3::core::Ellipsoid& ellipsoid, const double& thresholdGeo2rdr,
        const int& numiterGeo2rdr, const size_t& linesPerBlock,
        const double& demBlockMargin, const bool flatten)
{

    // number of bands in the input raster
    size_t nbands = inputRaster.numBands();
    std::cout << "nbands: " << nbands << std::endl;
    // create projection based on _epsg code
    std::unique_ptr<isce3::core::ProjectionBase> proj(
            isce3::core::createProj(geoGrid.epsg()));

    // Interpolator pointer
    auto interp = std::make_unique<
            isce3::core::Sinc2dInterpolator<std::complex<float>>>(
            isce3::core::SINC_LEN, isce3::core::SINC_SUB);

    // Compute number of blocks in the output geocoded grid
    size_t nBlocks = (geoGrid.length() + linesPerBlock - 1) / linesPerBlock;

    std::cout << "nBlocks: " << nBlocks << std::endl;
    // loop over the blocks of the geocoded Grid
    for (size_t block = 0; block < nBlocks; ++block) {
        std::cout << "block: " << block << std::endl;
        // Get block extents (of the geocoded grid)
        size_t lineStart, geoBlockLength;
        lineStart = block * linesPerBlock;
        if (block == (nBlocks - 1)) {
            geoBlockLength = geoGrid.length() - lineStart;
        } else {
            geoBlockLength = linesPerBlock;
        }
        size_t blockSize = geoBlockLength * geoGrid.width();

        // get a DEM interpolator for a block of DEM for the current geocoded
        // grid
        isce3::geometry::DEMInterpolator demInterp = isce3::geocode::loadDEM(
                demRaster, geoGrid, lineStart, geoBlockLength, geoGrid.width(),
                demBlockMargin);

        // X and Y indices (in the radar coordinates) for the
        // geocoded pixels (after geo2rdr computation)
        std::valarray<double> radarX(blockSize);
        std::valarray<double> radarY(blockSize);

        // First and last line of the data block in radar coordinates
        int azimuthFirstLine = radarGrid.length() - 1;
        int azimuthLastLine = 0;
        int rangeFirstPixel = radarGrid.width() - 1;
        int rangeLastPixel = 0;
        // First and last pixel of the data block in radar coordinates

        size_t geoGridWidth = geoGrid.width();
// Loop over lines, samples of the output grid
#pragma omp parallel for reduction(min                                    \
                                   : azimuthFirstLine,                    \
                                     rangeFirstPixel)                     \
        reduction(max                                                     \
                  : azimuthLastLine, rangeLastPixel)
        for (size_t kk = 0; kk < geoBlockLength * geoGridWidth; ++kk) {

            size_t blockLine = kk / geoGridWidth;
            size_t pixel = kk % geoGridWidth;

            // Global line index
            const size_t line = lineStart + blockLine;

            // y coordinate in the out put grid
	    // Assuming geoGrid.startY() and geoGrid.startX() represent the top-left
	    // corner of the first pixel, then 0.5 pixel shift is needed to get
	    // to the center of each pixel
            double y = geoGrid.startY() + geoGrid.spacingY() * (line + 0.5);

            // x in the output geocoded Grid
            double x = geoGrid.startX() + geoGrid.spacingX() * (pixel + 0.5);

            // compute the azimuth time and slant range for the
            // x,y coordinates in the output grid
            double aztime, srange;
            aztime = radarGrid.sensingMid();

            // coordinate in the output projection system
            const isce3::core::Vec3 xyz {x, y, 0.0};

            // transform the xyz in the output projection system to llh
            isce3::core::Vec3 llh = proj->inverse(xyz);

            // interpolate the height from the DEM for this pixel
            llh[2] = demInterp.interpolateLonLat(llh[0], llh[1]);

            // Perform geo->rdr iterations
            int geostat = isce3::geometry::geo2rdr(
                    llh, ellipsoid, orbit, imageGridDoppler, aztime, srange,
                    radarGrid.wavelength(), radarGrid.lookSide(),
                    thresholdGeo2rdr, numiterGeo2rdr, 1.0e-8);

            // Check convergence
            if (geostat == 0) {
                continue;
            }

            // get the row and column index in the radar grid
            double rdrY = (aztime - radarGrid.sensingStart()) * radarGrid.prf();

            double rdrX = (srange - radarGrid.startingRange()) /
                          radarGrid.rangePixelSpacing();

            if (rdrY < 0 || rdrX < 0 || rdrY >= radarGrid.length() ||
                rdrX >= radarGrid.width() ||
                not nativeDoppler.contains(aztime, srange))
                continue;

            azimuthFirstLine = std::min(
                    azimuthFirstLine, static_cast<int>(std::floor(rdrY)));
            azimuthLastLine =
                    std::max(azimuthLastLine,
                             static_cast<int>(std::ceil(rdrY) - 1));
            rangeFirstPixel = std::min(rangeFirstPixel,
                                            static_cast<int>(std::floor(rdrX)));
            rangeLastPixel = std::max(
                    rangeLastPixel, static_cast<int>(std::ceil(rdrX) - 1));

            // store the adjusted X and Y indices
            radarX[blockLine * geoGrid.width() + pixel] = rdrX;
            radarY[blockLine * geoGrid.width() + pixel] = rdrY;

        } // end loops over lines and pixel of output grid

        // Extra margin for interpolation to avoid gaps between blocks in output
        int interp_margin = 5;

        // Get min and max swath extents from among all threads
        azimuthFirstLine = std::max(azimuthFirstLine - interp_margin, 0);
        rangeFirstPixel = std::max(rangeFirstPixel - interp_margin, 0);

        azimuthLastLine = std::min(azimuthLastLine + interp_margin,
                                   static_cast<int>(radarGrid.length() - 1));
        rangeLastPixel = std::min(rangeLastPixel + interp_margin,
                                  static_cast<int>(radarGrid.width() - 1));

        if (azimuthFirstLine > azimuthLastLine ||
            rangeFirstPixel > rangeLastPixel)
            continue;

        // shape of the required block of data in the radar coordinates
        size_t rdrBlockLength = azimuthLastLine - azimuthFirstLine + 1;
        size_t rdrBlockWidth = rangeLastPixel - rangeFirstPixel + 1;

        // define the matrix based on the rasterbands data type
        isce3::core::Matrix<std::complex<float>> rdrDataBlock(rdrBlockLength,
                                                             rdrBlockWidth);
        isce3::core::Matrix<std::complex<float>> geoDataBlock(geoBlockLength,
                                                             geoGrid.width());

        // fill both matrices with zero
        rdrDataBlock.zeros();
        geoDataBlock.zeros();

        // for each band in the input:
        for (size_t band = 0; band < nbands; ++band) {

            std::cout << "band: " << band << std::endl;
            // get a block of data
            std::cout << "get data block " << std::endl;
            inputRaster.getBlock(rdrDataBlock.data(), rangeFirstPixel,
                                 azimuthFirstLine, rdrBlockWidth,
                                 rdrBlockLength, band + 1);

            // interpolate the data in radar grid to the geocoded grid.
            isce3::geocode::interpolate(rdrDataBlock, geoDataBlock, radarX,
                    radarY, azimuthFirstLine, rangeFirstPixel, interp.get(),
                    radarGrid, nativeDoppler, flatten);
            // set output
            std::cout << "set output " << std::endl;
            outputRaster.setBlock(geoDataBlock.data(), 0, lineStart,
                                  geoGrid.width(), geoBlockLength, band + 1);
        }
        // set output block of data
    } // end loop over block of output grid
}
