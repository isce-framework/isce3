//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2019-

#include "Geocode.h"

#include <isce/core/Basis.h>
#include <isce/core/Projections.h>
#include "DEMInterpolator.h"

using isce::core::Vec3;

template<class T>
void isce::geometry::Geocode<T>::
geocode(isce::io::Raster & inputRaster,
        isce::io::Raster & outputRaster,
        isce::io::Raster & demRaster) 
{

    // number of bands in the input raster
    size_t nbands = inputRaster.numBands();

    // create projection based on _epsg code
    isce::core::ProjectionBase * proj = isce::core::createProj(_epsgOut);

    // instantiate the DEMInterpolator 
    isce::geometry::DEMInterpolator demInterp;

    // Compute number of blocks in the output geocoded grid
    size_t nBlocks = _geoGridLength / _linesPerBlock;
    if ((_geoGridLength % _linesPerBlock) != 0)
        nBlocks += 1;

    std::cout << " nBlocks: " << nBlocks << std::endl;
    //loop over the blocks of the geocoded Grid
    for (size_t block = 0; block < nBlocks; ++block) {
        std::cout << "block : " << block << std::endl;
        // Get block extents (of the geocoded grid)
        size_t lineStart, geoBlockLength;
        lineStart = block * _linesPerBlock;
        if (block == (nBlocks - 1)) {
            geoBlockLength = _geoGridLength - lineStart;
        } else {
            geoBlockLength = _linesPerBlock;
        }
        size_t blockSize = geoBlockLength * _geoGridWidth;

        //First and last line of the data block in radar coordinates
        int azimuthFirstLine, azimuthLastLine;

        //First and last pixel of the data block in radar coordinates
        int rangeFirstPixel, rangeLastPixel;

        // load a block of DEM for the current geocoded grid
        _loadDEM(demRaster, demInterp, proj,
                lineStart, geoBlockLength, _geoGridWidth, 
                _demBlockMargin);

        //Given the current block on geocoded grid,
        //compute the bounding box of a block of data in the radar image.
        //This block of data will be used to interpolate the
        //values to the geocoded block
        _computeRangeAzimuthBoundingBox(lineStart, 
                        geoBlockLength, _geoGridWidth,
                        _radarBlockMargin, demInterp, proj,
                        azimuthFirstLine, azimuthLastLine,
                        rangeFirstPixel, rangeLastPixel);

        // shape of the required block of data in the radar coordinates
        size_t rdrBlockLength = azimuthLastLine - azimuthFirstLine + 1;
        size_t rdrBlockWidth = rangeLastPixel - rangeFirstPixel + 1;

        // X and Y indices (in the radar coordinates) for the 
        // geocoded pixels (after geo2rdr computation)
        std::valarray<double> radarX(blockSize);
    	std::valarray<double> radarY(blockSize);

        // Loop over lines of the output grid
        for (size_t blockLine = 0; blockLine < geoBlockLength; ++blockLine) {
            // Global line index
            const size_t line = lineStart + blockLine;
           
            // y coordinate in the out put grid
            double y = _geoGridStartY + _geoGridSpacingY*line;

            // Loop over geocoded grid pixels
            #pragma omp parallel for
            for (size_t pixel = 0; pixel < _geoGridWidth; ++pixel) {
                
                // x in the output geocoded Grid
                double x = _geoGridStartX + _geoGridSpacingX*pixel;

                // Consistency check
                
                // compute the azimuth time and slant range for the 
                // x,y coordinates in the output grid
                double aztime, srange;
                _geo2rdr(x, y, aztime, srange, demInterp, proj);

                // get the row and column index in the radar grid
                double rdrX, rdrY;
                rdrY = (aztime - _radarGrid.sensingStart()) *
                       (_radarGrid.prf() / _radarGrid.numberAzimuthLooks());
		
                rdrX = (srange - _radarGrid.startingRange()) /
                       (_radarGrid.numberRangeLooks() * _radarGrid.rangePixelSpacing());	

                // adjust the row and column indicies for the current block, 
                // i.e., moving the origin to the top-left of this radar block.
                rdrY -= azimuthFirstLine;
                rdrX -= rangeFirstPixel;
                
                //store the adjusted X and Y indices 
                radarX[blockLine*_geoGridWidth + pixel] = rdrX;
                radarY[blockLine*_geoGridWidth + pixel] = rdrY;

            } // end loop over pixels of output grid 
        } // end loops over lines of output grid

        // define the matrix based on the rasterbands data type
        isce::core::Matrix<T> rdrDataBlock(rdrBlockLength, rdrBlockWidth);        
        isce::core::Matrix<T> geoDataBlock(geoBlockLength, _geoGridWidth);

        // fill both matrices with zero
        rdrDataBlock.zeros();
        geoDataBlock.zeros();
         
        //for each band in the input:
        for (size_t band = 0; band < nbands; ++band){

            std::cout << "band: " << band << std::endl;
            // get a block of data
            std::cout << "get data block " << std::endl;
            inputRaster.getBlock(rdrDataBlock.data(),
                                rangeFirstPixel, azimuthFirstLine,
                                rdrBlockWidth, rdrBlockLength, band+1);

            // interpolate the data in radar grid to the geocoded grid
            std::cout << "interpolate " << std::endl;
            _interpolate(rdrDataBlock, geoDataBlock, radarX, radarY, 
                                rdrBlockWidth, rdrBlockLength);

            // set output
            std::cout << "set output " << std::endl;
            outputRaster.setBlock(geoDataBlock.data(), 0, lineStart, 
                                _geoGridWidth, geoBlockLength, band+1);
        }
        // set output block of data
    } // end loop over block of output grid

    outputRaster.setGeoTransform(_geoTrans);
    outputRaster.setEPSG(_epsgOut);
}

template<class T>
void isce::geometry::Geocode<T>::
_interpolate(isce::core::Matrix<T>& rdrDataBlock, 
            isce::core::Matrix<T>& geoDataBlock,
            std::valarray<double>& radarX, std::valarray<double>& radarY, 
            int radarBlockWidth, int radarBlockLength)
{

    size_t length = geoDataBlock.length();
    size_t width = geoDataBlock.width();
    double extraMargin = 4.0;

    #pragma omp parallel for
    for (size_t kk = 0; kk < length*width; ++kk) {
        
        size_t i = kk / width;
        size_t j = kk % width;

        if (radarX[i*width + j] >= extraMargin &&
                    radarY[i*width + j] >= extraMargin &&
                    radarX[i*width + j] < (radarBlockWidth - extraMargin) &&
                    radarY[i*width + j] < (radarBlockLength - extraMargin) ) {

                geoDataBlock(i,j) = _interp->interpolate(radarX[i*width + j],
                                                radarY[i*width + j], rdrDataBlock);

        }

    }
}

template<class T>
void isce::geometry::Geocode<T>::
_loadDEM(isce::io::Raster demRaster,
        isce::geometry::DEMInterpolator & demInterp,
        isce::core::ProjectionBase * proj, 
        int lineStart, int blockLength, 
        int blockWidth, double demMargin)
{

    //Create projection for DEM
    int epsgcode = demRaster.getEPSG();
    
    //Initialize bounds
    double minX = -1.0e64;
    double maxX = 1.0e64;
    double minY = -1.0e64;
    double maxY = 1.0e64;

    //Projection systems are different
    if (epsgcode != proj->code())
    {
        
        //Create transformer to match the DEM 
        isce::core::ProjectionBase *demproj = isce::core::createProj(epsgcode);

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
            Vec3 outpt = { _geoGridStartX + _geoGridSpacingX * pixInd[i],
                           _geoGridStartY + _geoGridSpacingY * lineInd[i],
                           0.0};

            Vec3 dempt;
            if (!projTransform(proj, demproj, outpt, dempt))
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
        maxY = _geoGridStartY + _geoGridSpacingY*lineStart;
        minY = _geoGridStartY + _geoGridSpacingY*(lineStart + blockLength - 1);
        minX = _geoGridStartX;
        maxX = _geoGridStartX + _geoGridSpacingX*(blockWidth - 1);
    }

    //If not LonLat, scale to meters
    demMargin = (epsgcode != 4326)? isce::core::decimaldeg2meters(demMargin) : demMargin;

    // Account for margins
    minX -= demMargin;
    maxX += demMargin;
    minY -= demMargin;
    maxY += demMargin;

    // load the DEM for this bounding box
    demInterp.loadDEM(demRaster, minX, maxX, minY, maxY);

    if (demInterp.width() == 0 || demInterp.length() == 0)
        std::cout << "warning there are not enough DEM coverage in the bounding box. " << std::endl;

    // declare the dem interpolator
    demInterp.declare();
}

template<class T>
void isce::geometry::Geocode<T>::
_computeRangeAzimuthBoundingBox(int lineStart, int blockLength, int blockWidth,
                        int margin, isce::geometry::DEMInterpolator & demInterp,
                        isce::core::ProjectionBase * proj,
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

    // Cache the multilooked radar grid length and width
    const size_t rgLength = _radarGrid.length() / _radarGrid.numberAzimuthLooks();
    const size_t rgWidth = _radarGrid.width() / _radarGrid.numberRangeLooks();

    // Cache the multilooked radar grid spacing
    const double dtaz = _radarGrid.numberAzimuthLooks() / _radarGrid.prf();
    const double dtrg = _radarGrid.numberRangeLooks() * _radarGrid.rangePixelSpacing();

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

    // compute geo2rdr for the 4 corners
    for (size_t i = 0; i<4; ++i){
        _geo2rdr(X[i], Y[i], azimuthTime[i], slantRange[i], demInterp, proj); 
    }

    // the first azimuth line
    azimuthFirstLine = (azimuthTime.min() - _radarGrid.sensingStart()) / dtaz;

    // the last azimuth line
    azimuthLastLine = (azimuthTime.max() - _radarGrid.sensingStart()) / dtaz;

    // the first and last range pixels 
    rangeFirstPixel = (slantRange.min() - _radarGrid.startingRange()) / dtrg;
    rangeLastPixel = (slantRange.max() - _radarGrid.startingRange()) / dtrg;

    // extending the radar bounding box by the extra margin
    azimuthFirstLine -= margin;
    azimuthLastLine  += margin;
    rangeFirstPixel -= margin;
    rangeLastPixel  += margin;

    // make sure the radar block's bounding box is inside the existing radar grid
    if (azimuthFirstLine < 0)
        azimuthFirstLine = 0;

    if (azimuthLastLine > (rgLength - 1))
        azimuthLastLine = rgLength - 1;

    if (rangeFirstPixel < 0)
        rangeFirstPixel = 0;

    if (rangeLastPixel > (rgWidth - 1))
        rangeLastPixel = rgWidth - 1;

}

template<class T>
void isce::geometry::Geocode<T>::
_geo2rdr(double x, double y, 
        double & azimuthTime, double & slantRange,
        isce::geometry::DEMInterpolator & demInterp,
        isce::core::ProjectionBase * proj)
{
    // coordinate in the output projection system
    const Vec3 xyz{x, y, 0.0};

    // transform the xyz in the output projection system to llh
    Vec3 llh = proj->inverse(xyz);

    // interpolate the height from the DEM for this pixel
    llh[2] = demInterp.interpolateLonLat(llh[0], llh[1]);

    // Perform geo->rdr iterations
    int geostat = isce::geometry::geo2rdr(
                    llh, _ellipsoid, _orbit, _doppler,
                    azimuthTime, slantRange, _radarGrid.wavelength(), _threshold,
                    _numiter, 1.0e-8);

    // Check convergence
    if (geostat == 0) {
        azimuthTime = 0.0;
        slantRange = -1.0e-16;
        return;
    }

    // Compute TCN basis for geo2rdr solution for checking side consistency
    Vec3 xyzsat, velsat;
    _orbit.interpolate(azimuthTime, xyzsat, velsat, isce::core::HERMITE_METHOD);
    const isce::core::Basis tcn(xyzsat, velsat);

    // Check the side of the C-component of the look vector to the ground point
    const Vec3 targxyz = _ellipsoid.lonLatToXyz(llh);
    const Vec3 deltaxyz = targxyz - xyzsat;
    const double targ_c = deltaxyz.dot(tcn.x1());

    // Positive values are invalid
    if ((targ_c * _lookSide > 0.0)) {
        azimuthTime = 0.0;
        slantRange = -1.0e-16;
    }

}

template class isce::geometry::Geocode<float>;
template class isce::geometry::Geocode<double>;
template class isce::geometry::Geocode<std::complex<float>>;
template class isce::geometry::Geocode<std::complex<double>>;

