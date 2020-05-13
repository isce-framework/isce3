#include "geocodeSlc.h"

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
        const int sincLength,
        const bool flatten) {

    // number of bands in the input raster
    size_t nbands = inputRaster.numBands();
    std::cout << "nbands: "<< nbands << std::endl;
    // create projection based on _epsg code
    isce::core::ProjectionBase * proj = isce::core::createProj(geoGrid.epsg());
    
    // instantiate the DEMInterpolator
    isce::geometry::DEMInterpolator demInterp;

    std::cout << "instatntiate the interpolator" << std::endl;
    // Interpolator pointer
    isce::core::Interpolator<std::complex<float>> * interp;
    interp = new isce::core::Sinc2dInterpolator<std::complex<float>>(
                    sincLength-1, isce::core::SINC_SUB);

    std::cout << "interpolator instantiated" << std::endl;
    // Compute number of blocks in the output geocoded grid
    size_t nBlocks = geoGrid.length() / linesPerBlock;
    if ((geoGrid.length() % linesPerBlock) != 0)
        nBlocks += 1;

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

        // load a block of DEM for the current geocoded grid
        isce::geocode::loadDEM(demRaster, demInterp, proj, geoGrid,
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

        #pragma omp parallel shared(azimuthFirstLine, rangeFirstPixel, azimuthLastLine, rangeLastPixel)
        {
            // Init thread-local swath extents
            int localAzimuthFirstLine = radarGrid.length() - 1;
            int localAzimuthLastLine = 0;
            int localRangeFirstPixel = radarGrid.width() - 1;
            int localRangeLastPixel = 0;

            // Loop over lines, samples of the output grid
            #pragma omp for collapse(2)
            for (size_t blockLine = 0; blockLine < geoBlockLength; ++blockLine) {
                for (size_t pixel = 0; pixel < geoGrid.width(); ++pixel) {

                    // numDone++;

                    // Global line index
                    const size_t line = lineStart + blockLine;

                    // y coordinate in the out put grid
                    double y = geoGrid.startY() + geoGrid.spacingY() * line;

                    // x in the output geocoded Grid
                    double x = geoGrid.startX() + geoGrid.spacingX() * pixel;

                    // Consistency check

                    // compute the azimuth time and slant range for the
                    // x,y coordinates in the output grid
                    double aztime, srange;
                    isce::geocode::geo2rdr(x, y, aztime, srange, demInterp, proj, 
                            orbit, imageGridDoppler, ellipsoid, 
                            radarGrid.wavelength(), radarGrid.lookSide(),
                            thresholdGeo2rdr, numiterGeo2rdr);

                    if (std::isnan(aztime) || std::isnan(srange))
                        continue;

                    // get the row and column index in the radar grid
                    double rdrX, rdrY;
                    rdrY = (aztime - radarGrid.sensingStart()) * radarGrid.prf();

                    rdrX = (srange - radarGrid.startingRange()) / radarGrid.rangePixelSpacing();

                    if (rdrY < 0 || rdrX < 0 || rdrY >= radarGrid.length() ||
                        rdrX >= radarGrid.width())
                        continue;

                    localAzimuthFirstLine = std::min(localAzimuthFirstLine, (int)std::floor(rdrY));
                    localAzimuthLastLine = std::max(localAzimuthLastLine, (int)std::ceil(rdrY) - 1);
                    localRangeFirstPixel = std::min(localRangeFirstPixel, (int)std::floor(rdrX));
                    localRangeLastPixel = std::max(localRangeLastPixel, (int)std::ceil(rdrX) - 1);

                    //store the adjusted X and Y indices
                    radarX[blockLine * geoGrid.width() + pixel] = rdrX;
                    radarY[blockLine * geoGrid.width() + pixel] = rdrY;
                    
                    // doppler to be added back after interpolation
                    double phase = nativeDoppler.eval(aztime, srange) * 2*M_PI*aztime;
                    //

                    if (flatten) {
                       phase += (4.0 * (M_PI/radarGrid.wavelength())) * srange;
                    }

                    phase = modulo_f(phase, 2.0*M_PI);
                    const std::complex<double> cpxPhase(std::cos(phase), std::sin(phase));

                    geometricalPhase[blockLine * geoGrid.width() + pixel] = cpxPhase;
                    //radarX[blockLine * geoGrid.width() + pixel] = srange;
                    //radarY[blockLine * geoGrid.width() + pixel] = aztime;

                } // end loop over pixels of output grid
            } // end loops over lines of output grid

            #pragma omp critical
            {
                // Get min and max swath extents from among all threads
                azimuthFirstLine = std::min(azimuthFirstLine, localAzimuthFirstLine);
                azimuthLastLine = std::max(azimuthLastLine, localAzimuthLastLine);
                rangeFirstPixel = std::min(rangeFirstPixel, localRangeFirstPixel);
                rangeLastPixel = std::max(rangeLastPixel, localRangeLastPixel);
            }
        }

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
                         azimuthFirstLine, rangeFirstPixel, interp);

            // set output
            std::cout << "set output " << std::endl;
            outputRaster.setBlock(geoDataBlock.data(), 0, lineStart,
                                  geoGrid.width(), geoBlockLength, band + 1);
        }
        // set output block of data
    } // end loop over block of output grid

    outputRaster.setGeoTransform(geoGrid.geotransform());
    outputRaster.setEPSG(geoGrid.epsg());
}


