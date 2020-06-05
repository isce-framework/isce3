#include "geocodeSlc.h"
#include <memory>

void isce::geocode::geocodeSlc(isce::io::Raster & outputRaster,
        isce::io::Raster & inputRaster,
        isce::io::Raster & demRaster,
        const isce::product::RadarGridParameters & radarGrid,
        const isce::product::GeoGridParameters & geoGrid,
        const isce::core::Orbit& orbit, 
        const isce::core::LUT2d<double>& nativeDoppler,
        const isce::core::LUT2d<double>& imageGridDoppler,
        const isce::core::Ellipsoid & ellipsoid,
        const double & thresholdGeo2rdr, 
        const int & numiterGeo2rdr,
        const size_t & linesPerBlock,
        const double & demBlockMargin,
        const bool flatten) {

    // number of bands in the input raster
    size_t nbands = inputRaster.numBands();
    std::cout << "nbands: "<< nbands << std::endl;
    // create projection based on _epsg code
    std::unique_ptr<isce::core::ProjectionBase> proj(
                isce::core::createProj(geoGrid.epsg()));

    // Interpolator pointer
    auto interp = std::make_unique<isce::core::Sinc2dInterpolator<
                        std::complex<float>>>(isce::core::SINC_LEN, isce::core::SINC_SUB);

    // Compute number of blocks in the output geocoded grid
    size_t nBlocks = (geoGrid.length() + linesPerBlock - 1) / linesPerBlock;

    std::cout << "nBlocks: " << nBlocks << std::endl;
    //loop over the blocks of the geocoded Grid
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

        //First and last line of the data block in radar coordinates
        int azimuthFirstLine = radarGrid.length()-1;
        int azimuthLastLine = 0;

        //First and last pixel of the data block in radar coordinates
        int rangeFirstPixel = radarGrid.width()-1;
        int rangeLastPixel = 0;

        // get a DEM interpolator for a block of DEM for the current geocoded grid
        isce::geometry::DEMInterpolator demInterp = isce::geocode::loadDEM(
                demRaster, geoGrid,
                lineStart, geoBlockLength, geoGrid.width(),
                demBlockMargin);

        std::cout << "DEM loaded" << std::endl;
        // X and Y indices (in the radar coordinates) for the
        // geocoded pixels (after geo2rdr computation)
        std::valarray<double> radarX(blockSize);
        std::valarray<double> radarY(blockSize);

        // container for the sum of the carrier phase (Doppler) to be added back and 
        // the geometrical phase to be removed for flattening the SLC phase.
        std::valarray<std::complex<double>> geometricalPhase(blockSize);

        int localAzimuthFirstLine = radarGrid.length() - 1;
        int localAzimuthLastLine = 0;
        int localRangeFirstPixel = radarGrid.width() - 1;
        int localRangeLastPixel = 0;

        size_t geoGridWidth = geoGrid.width();
        // Loop over lines, samples of the output grid
        #pragma omp parallel for reduction(min:localAzimuthFirstLine,localRangeFirstPixel) reduction(max:localAzimuthLastLine,localRangeLastPixel)
            for (size_t kk = 0; kk < geoBlockLength * geoGridWidth; ++kk) {

                size_t blockLine = kk / geoGridWidth;
                size_t pixel = kk % geoGridWidth;

                    // Global line index
                    const size_t line = lineStart + blockLine;

                    // y coordinate in the out put grid
                    double y = geoGrid.startY() + geoGrid.spacingY() * line;

                    // x in the output geocoded Grid
                    double x = geoGrid.startX() + geoGrid.spacingX() * pixel;

                    // compute the azimuth time and slant range for the
                    // x,y coordinates in the output grid
                    double aztime, srange;
                    aztime = radarGrid.sensingMid();

                    // coordinate in the output projection system
                    const isce::core::Vec3 xyz{x, y, 0.0};

                    // transform the xyz in the output projection system to llh
                    isce::core::Vec3 llh = proj->inverse(xyz);

                    // interpolate the height from the DEM for this pixel
                    llh[2] = demInterp.interpolateLonLat(llh[0], llh[1]);

                    // Perform geo->rdr iterations
                    int geostat = isce::geometry::geo2rdr(
                            llh, ellipsoid, orbit, imageGridDoppler,
                            aztime, srange, radarGrid.wavelength(),
                            radarGrid.lookSide(), thresholdGeo2rdr, numiterGeo2rdr, 1.0e-8);

                    // Check convergence
                    if (geostat == 0) {
                        aztime = std::numeric_limits<double>::quiet_NaN();
                        srange = std::numeric_limits<double>::quiet_NaN();
                    }

                    if (std::isnan(aztime) || std::isnan(srange))
                        continue;

                    // get the row and column index in the radar grid
                    double rdrY = (aztime - radarGrid.sensingStart()) * radarGrid.prf();

                    double rdrX = (srange - radarGrid.startingRange()) / radarGrid.rangePixelSpacing();

                    if (rdrY < 0 || rdrX < 0 || rdrY >= radarGrid.length() ||
                        rdrX >= radarGrid.width())
                        continue;

                    localAzimuthFirstLine = std::min(localAzimuthFirstLine, static_cast<int>(std::floor(rdrY)));
                    localAzimuthLastLine = std::max(localAzimuthLastLine, static_cast<int>(std::ceil(rdrY) - 1));
                    localRangeFirstPixel = std::min(localRangeFirstPixel, static_cast<int>(std::floor(rdrX)));
                    localRangeLastPixel = std::max(localRangeLastPixel, static_cast<int>(std::ceil(rdrX) - 1));

                    //store the adjusted X and Y indices
                    radarX[blockLine * geoGrid.width() + pixel] = rdrX;
                    radarY[blockLine * geoGrid.width() + pixel] = rdrY;
                    
                    // doppler to be added back after interpolation
                    double phase = nativeDoppler.eval(aztime, srange) * 2*M_PI*aztime;
                    //

                    if (flatten) {
                       phase += (4.0 * (M_PI/radarGrid.wavelength())) * srange;
                    }

                    const std::complex<double> cpxPhase(std::cos(phase), std::sin(phase));

                    geometricalPhase[blockLine * geoGrid.width() + pixel] = cpxPhase;

            } // end loops over lines and pixel of output grid

        // Get min and max swath extents from among all threads
        azimuthFirstLine = std::min(azimuthFirstLine, localAzimuthFirstLine);
        azimuthLastLine = std::max(azimuthLastLine, localAzimuthLastLine);
        rangeFirstPixel = std::min(rangeFirstPixel, localRangeFirstPixel);
        rangeLastPixel = std::max(rangeLastPixel, localRangeLastPixel);

        if (azimuthFirstLine > azimuthLastLine || rangeFirstPixel > rangeLastPixel)
            continue;

        // shape of the required block of data in the radar coordinates
        size_t rdrBlockLength = azimuthLastLine - azimuthFirstLine + 1;
        size_t rdrBlockWidth = rangeLastPixel - rangeFirstPixel + 1;

        // define the matrix based on the rasterbands data type
        isce::core::Matrix<std::complex<float>> rdrDataBlock(rdrBlockLength, rdrBlockWidth);
        isce::core::Matrix<std::complex<float>> geoDataBlock(geoBlockLength, geoGrid.width());

        // fill both matrices with zero
        rdrDataBlock.zeros();
        geoDataBlock.zeros();

        //for each band in the input:
        for (size_t band = 0; band < nbands; ++band) {

            std::cout << "band: " << band << std::endl;
            // get a block of data
            std::cout << "get data block " << std::endl;
            inputRaster.getBlock(rdrDataBlock.data(),
                                 rangeFirstPixel, azimuthFirstLine,
                                 rdrBlockWidth, rdrBlockLength, band + 1);

            //baseband the SLC in the radar grid
            const double blockStartingRange = radarGrid.startingRange() + 
                                    rangeFirstPixel * radarGrid.rangePixelSpacing();
            const double blockSensingStart = radarGrid.sensingStart() + 
                                    azimuthFirstLine / radarGrid.prf();

            isce::geocode::baseband(rdrDataBlock, blockStartingRange, blockSensingStart,
                    radarGrid.rangePixelSpacing(), radarGrid.prf(),
                    nativeDoppler);

            // interpolate the data in radar grid to the geocoded grid. 
            // Also the geometrical phase, which is the phase of the carrier 
            // to be added back and the geometrical phase to be removed is applied.
            std::cout << "resample " << std::endl;
            isce::geocode::interpolate(rdrDataBlock, geoDataBlock, 
                        radarX, radarY, geometricalPhase,
                         rdrBlockWidth, rdrBlockLength,
                         azimuthFirstLine, rangeFirstPixel, interp.get());

            // set output
            std::cout << "set output " << std::endl;
            outputRaster.setBlock(geoDataBlock.data(), 0, lineStart,
                                  geoGrid.width(), geoBlockLength, band + 1);
        }
        // set output block of data
    } // end loop over block of output grid
}


