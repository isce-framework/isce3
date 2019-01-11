//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2019-

#include "Geocode.h"

void isce::geometry::Geocode::
geocode(isce::io::Raster & inputRaster,
        isce::io::Raster & outputRaster,
        isce::io::Raster & demRaster) 
{

    // create projection based on _epsg code
     _proj = isce::core::createProj(_epsgOut);

    // instantiate the DEMInterpolator 
    isce::geometry::DEMInterpolator demInterp;

    // Compute number of blocks in the output geocoded grid
    size_t nBlocks = _geoGridLength / _linesPerBlock;
    if ((_geoGridLength % _linesPerBlock) != 0)
        nBlocks += 1;

    //loop over the blocks of DEM
    for (size_t block = 0; block < nBlocks; ++block) {

        // Get block extents (of the geocoded grid)
        size_t lineStart, blockLength;
        lineStart = block * _linesPerBlock;
        if (block == (nBlocks - 1)) {
            blockLength = _geoGridLength - lineStart;
        } else {
            blockLength = _linesPerBlock;
        }
        size_t blockSize = blockLength * _geoGridWidth;

        //First and last line of the data block in radar coordinates
        int azimuthFirstLine, azimuthLastLine;

        //First and last pixel of the data block in radar coordinates
        int rangeFirstPixel, rangeLastPixel;

        // load a block of DEM for the current geocoded grid
        _loadDEM(demRaster, demInterp, _proj,
                lineStart, blockLength, _geoGridWidth, 
                _demBlockMargin);

        //Given the current block on geocoded grid,
        //compute the bounding box of a block of data in the radar image.
        //This block of data will be used to interpolate the
        //values to the geocoded block

        _computeRangeAzimuthBoundingBox(lineStart, 
                        blockLength, _geoGridWidth,
                        _radarBlockMargin, demInterp,
                        azimuthFirstLine, azimuthLastLine,
                        rangeFirstPixel, rangeLastPixel);

        size_t rdrBlockLength = azimuthLastLine - azimuthFirstLine + 1;
        size_t rdrBlockWidth = rangeLastPixel - rangeFirstPixel + 1;
        size_t rdrBlockSize = rdrBlockLength * rdrBlockWidth;

        // X and Y indicies of the geocoded pixels in the radar coordinates
        std::valarray<double> radarX(blockSize);
    	std::valarray<double> radarY(blockSize);

        // Loop over lines of the output grid
        for (size_t blockLine = 0; blockLine < blockLength; ++blockLine) {

            // Global line index
            const size_t line = lineStart + blockLine;
           
            // y coordinate in the out put grid
            double y = _geoGridStartY + _geoGridSpacingY*line;

            // Loop over DEM pixels
            #pragma omp parallel for
            for (size_t pixel = 0; pixel < _geoGridWidth; ++pixel) {
                
                // x in the output geocoded Grid
                double x = _geoGridStartX + _geoGridSpacingX*pixel;
                
                // compute the azimuth time and slant range for the 
                // x,y coordinates in the output grid
                double aztime, srange;
                _geo2rdr(x, y, aztime, srange, demInterp);

                // get the row and column index in the radar grid
                double rdrX, rdrY;
                rdrY = (aztime - _azimuthStartTime.secondsSinceEpoch(_refEpoch))/
                                _azimuthTimeInterval;
		
                rdrX = (srange - _startingRange)/_rangeSpacing;		

                // adjust the x,y indicies for the current block, 
                // i.e., moving the origin to the top-left of this radar block.
                rdrY -= azimuthFirstLine;
                rdrX -= rangeFirstPixel;

                // Check if solution is out of bounds
                /*bool isOutside = false;
                if ((aztime < t0) || (aztime > tend))
                    isOutside = true;
                if ((slantRange < r0) || (slantRange > rngend))
                    isOutside = true;
                */

                //if (!isOutside) {
                    // interpolate the inputRaster to get the value at aztime, slantRange
                radarX[blockLine*_geoGridWidth + pixel] = rdrX;
                radarY[blockLine*_geoGridWidth + pixel] = rdrY;
                //} else {
                //    output[index] = NULL_VALUE;
                //}

            } // end loop over pixels of output grid 
        } // end loops over lines of output grid
        
        // define the matrix based on the rasterbands data type
        //isce::core::Matrix<T> rdrDataBlock(rdrBlockLength, rdrBlockWidth);
        isce::core::Matrix<float> rdrDataBlock(rdrBlockLength, rdrBlockWidth);

        isce::core::Matrix<float> geoDataBlock(blockLength, _geoGridWidth);

        //for each band in the input:
	// get a block of data
        inputRaster.getBlock(rdrDataBlock.data(),
                                    rangeFirstPixel, azimuthFirstLine,
                                    rdrBlockWidth, rdrBlockLength);
        
        _interpolate(rdrDataBlock, geoDataBlock, radarX, radarY);

        outputRaster.setBlock(geoDataBlock.data(), 0, lineStart, _geoGridWidth, blockLength);

        // set output block of data
    } // end loop over block of DEM

}

/*
void isce::geometry::Geocode<T>::
_interpolate(isce::core::Matrix<T> rdrDataBlock, isce::core::Matrix<T> geoDataBlock,
            std::valarray<double> radarX, std::valarray<double> radarY);

*/

void isce::geometry::Geocode::
_interpolate(isce::core::Matrix<float> rdrDataBlock, isce::core::Matrix<float> geoDataBlock,
        std::valarray<double> radarX, std::valarray<double> radarY)
{
    size_t blockSize = radarX.size();
    #pragma omp parallel for
    for (size_t i = 0; i < blockSize; ++i)
        geoDataBlock[i] = _interp->interpolate(radarX[i], radarY[i], rdrDataBlock);

}


void isce::geometry::Geocode::
_loadDEM(isce::io::Raster demRaster,
        isce::geometry::DEMInterpolator & demInterp,
        isce::core::ProjectionBase * _proj, 
        int lineStart, int blockLength, 
        int blockWidth, double demMargin)
{
    // convert the corner of the current geocoded grid to lon lat
    double maxY = _geoGridStartY + _geoGridSpacingY*lineStart;
    double minY = _geoGridStartY + _geoGridSpacingY*(lineStart + blockLength - 1);
    double minX = _geoGridStartX;
    double maxX = _geoGridStartX + _geoGridSpacingX*(blockWidth - 1);

    isce::core::cartesian_t xyz;
    isce::core::cartesian_t llh;

    // top left corner of the box
    xyz[0] = minX;
    xyz[1] = maxY;
    _proj->inverse(xyz, llh);

    double minLon = llh[0];
    double maxLat = llh[1];

    // lower right corner of the box
    xyz[0] = maxX;
    xyz[1] = minY;
    _proj->inverse(xyz, llh);

    double maxLon = llh[0];
    double minLat = llh[1];

    // convert the margin to radians
    demMargin *= (M_PI/180.0);

    // Account for margins
    minLon -= demMargin;
    maxLon += demMargin;
    minLat -= demMargin;
    maxLat += demMargin;

    // load the DEM for this bounding box
    demInterp.loadDEM(demRaster, minLon, maxLon, minLat, maxLat,
                                    demRaster.getEPSG());

    // declare the dem interpolator
    demInterp.declare();

}

void isce::geometry::Geocode::
_computeRangeAzimuthBoundingBox(int lineStart, int blockLength, int blockWidth,
                        int margin, isce::geometry::DEMInterpolator & demInterp,
                        int & azimuthFirstLine, int & azimuthLastLine,
                        int & rangeFirstPixel, int & rangeLastPixel)
{
    // to store the four corner of the block on ground
    std::valarray<double> X(4);
    std::valarray<double> Y(4);

    // to store the azimuth time and slant range corresponding to 
    // the corner of the block on ground
    std::valarray<double> azimuthTime(4);
    std::valarray<double> slantRange(4);

    //top left corener on ground
    Y[0] = _geoGridStartY + _geoGridSpacingY*lineStart;
    X[0] = _geoGridStartX;

    //top right corener on ground
    Y[1] = _geoGridStartY + _geoGridSpacingY*lineStart;
    X[1] = _geoGridStartX + _geoGridSpacingX*(blockWidth - 1);

    //bottom left corener on ground 
    Y[2] = _geoGridStartY + _geoGridSpacingY*(lineStart + blockLength - 1);
    X[2] = _geoGridStartX;
    
    //bottom right corener on ground
    Y[3] = _geoGridStartY + _geoGridSpacingY*(lineStart + blockLength - 1);
    X[3] = _geoGridStartX + _geoGridSpacingX*(blockWidth - 1);


    for (size_t i = 0; i<4; ++i){
        _geo2rdr(X[i], Y[i], azimuthTime[i], slantRange[i], demInterp);    
    }

    azimuthFirstLine = (azimuthTime.min() - _azimuthStartTime.secondsSinceEpoch(_refEpoch))/
                                _azimuthTimeInterval;

    azimuthLastLine = (azimuthTime.max() - _azimuthStartTime.secondsSinceEpoch(_refEpoch))/
                                _azimuthTimeInterval;

    rangeFirstPixel = (slantRange.min() - _startingRange)/_rangeSpacing;
    rangeLastPixel = (slantRange.max() - _startingRange)/_rangeSpacing;

    azimuthFirstLine -= margin;
    azimuthLastLine  += margin;
    rangeFirstPixel -= margin;
    rangeLastPixel  += margin;

    if (azimuthFirstLine < 0)
        azimuthFirstLine = 0;

    if (azimuthLastLine > (_radarRasterLength - 1))
        azimuthLastLine = _radarRasterLength - 1;

    if (rangeFirstPixel < 0)
        rangeFirstPixel = 0;

    if (rangeLastPixel > (_radarRasterWidth - 1))
        rangeLastPixel = _radarRasterWidth - 1;

}


void isce::geometry::Geocode::
_geo2rdr(double x, double y, 
        double & azimuthTime, double & slantRange,
        isce::geometry::DEMInterpolator & demInterp)
{
    // coordinate in the output projection system
    isce::core::cartesian_t xyz{x, y, 0.0};

    // coordinate in lon lat height
    isce::core::cartesian_t llh;

    // transform the xyz in the output projection system to llh
    _proj->inverse(xyz, llh);

    // interpolate the height from the DEM for this pixel
    llh[2] = demInterp.interpolateLonLat(llh[0], llh[1]);

    // Perform geo->rdr iterations
    int geostat = isce::geometry::geo2rdr(
                    llh, _ellipsoid, _orbit, _doppler, _mode, azimuthTime, slantRange, _threshold,
                    _numiter, 1.0e-8);

}


