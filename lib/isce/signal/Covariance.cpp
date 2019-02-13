// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2018-
//

#include "Covariance.h"

template<class T>
void isce::signal::Covariance<T>::
covariance(std::map<std::string, isce::io::Raster> & slc,
            std::map<std::pair<std::string, std::string>, isce::io::Raster> & cov)

{

    size_t numPolarizations = slc.size();
    size_t numCovElements = cov.size();

    std::string coPol;
    std::string crossPol;

    if (numPolarizations == 1){
        std::cout << "Covariance estimation needs at least two polarizations" << std::endl;
    } else if (numPolarizations == 2 && numCovElements == 3) {

        _dualPol = true;

        if (slc.count("hh") > 0)
            _coPol = "hh";
        else if (slc.count("vv") > 0)
            _coPol = "vv";

        if (slc.count("hv") > 0)
            _crossPol = "hv";
        else if (slc.count("vh") > 0)
            _crossPol = "vh";

    } else if (numPolarizations == 4 && numCovElements == 10) {
        _quadPol = true;
    } 

    // instantiate the crossmul object
    isce::signal::Crossmul crsmul;

    // set up crossmul
    crsmul.rangeLooks(_rangeLooks);

    crsmul.azimuthLooks(_azimuthLooks);

    crsmul.doCommonAzimuthbandFiltering(false);

    crsmul.doCommonRangebandFiltering(false);   

    if (_dualPol) {

        crsmul.crossmul(slc[_coPol], slc[_coPol], cov[std::make_pair(_coPol, _coPol)]);

        crsmul.crossmul(slc[_coPol], slc[_crossPol], cov[std::make_pair(_coPol, _crossPol)]);

        crsmul.crossmul(slc[_crossPol], slc[_crossPol], cov[std::make_pair(_crossPol, _crossPol)]);


    } else if (_quadPol){

        crsmul.crossmul(slc["hh"], slc["hh"], cov[std::make_pair("hh", "hh")]);

        crsmul.crossmul(slc["hh"], slc["vh"], cov[std::make_pair("hh", "vh")]);

        crsmul.crossmul(slc["hh"], slc["hv"], cov[std::make_pair("hh", "hv")]);

        crsmul.crossmul(slc["hh"], slc["vv"], cov[std::make_pair("hh", "vv")]);

        crsmul.crossmul(slc["vh"], slc["vh"], cov[std::make_pair("vh", "vh")]);

        crsmul.crossmul(slc["vh"], slc["hv"], cov[std::make_pair("vh", "hv")]);

        crsmul.crossmul(slc["vh"], slc["vv"], cov[std::make_pair("vh", "vv")]);

        crsmul.crossmul(slc["hv"], slc["hv"], cov[std::make_pair("hv", "hv")]);

        crsmul.crossmul(slc["hv"], slc["vv"], cov[std::make_pair("hv", "vv")]);

        crsmul.crossmul(slc["vv"], slc["vv"], cov[std::make_pair("vv", "vv")]);

    }
}

template<class T>
void isce::signal::Covariance<T>::
geocodeCovariance(isce::io::Raster& rdrCov,
                isce::io::Raster& geoCov,
                isce::io::Raster & demRaster) {

    _correctRtcFlag = false;
    _correctOrientationFlag = false;

    isce::io::Raster rtcRaster("/vsimem/dummyRtc", 1, 1, 1, GDT_Float32, "ENVI");
    isce::io::Raster orientationAngleRaster("/vsimem/dummyOrient", 1, 1, 1, GDT_Float32, "ENVI");
    
    geocodeCovariance(rdrCov,
                geoCov,
                demRaster,
                rtcRaster,
                orientationAngleRaster);


}

template<class T>
void isce::signal::Covariance<T>::
geocodeCovariance(isce::io::Raster& rdrCov,
                isce::io::Raster& geoCov,
                isce::io::Raster & demRaster,
                isce::io::Raster& rtcRaster) {
    
    _correctOrientationFlag = false;
    isce::io::Raster orientationAngleRaster("/vsimem/dummyOrient", 1, 1, 1, GDT_Float32, "ENVI");
    
    geocodeCovariance(rdrCov,
                geoCov,
                demRaster,
                rtcRaster,
                orientationAngleRaster);

}

template<class T>
void isce::signal::Covariance<T>::
geocodeCovariance(isce::io::Raster& rdrCov,
                isce::io::Raster& geoCov,
                isce::io::Raster & demRaster,
                isce::io::Raster& rtc,
                isce::io::Raster& orientationAngleRaster) {

    // number of bands in the input raster
    size_t nbands = rdrCov.numBands();

    // create projection based on _epsg code
    _proj = isce::core::createProj(_epsgOut);

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
        _loadDEM(demRaster, demInterp, _proj,
                lineStart, geoBlockLength, _geoGridWidth, 
                _demBlockMargin);

        //Given the current block on geocoded grid,
        //compute the bounding box of a block of data in the radar image.
        //This block of data will be used to interpolate the
        //values to the geocoded block
        _computeRangeAzimuthBoundingBox(lineStart, 
                        geoBlockLength, _geoGridWidth,
                        _radarBlockMargin, demInterp,
                        azimuthFirstLine, azimuthLastLine,
                        rangeFirstPixel, rangeLastPixel);

        // shape of the required block of data in the radar coordinates
        size_t rdrBlockLength = azimuthLastLine - azimuthFirstLine + 1;
        size_t rdrBlockWidth = rangeLastPixel - rangeFirstPixel + 1;
        size_t rdrBlockSize = rdrBlockLength * rdrBlockWidth;

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

            // Loop over DEM pixels
            //#pragma omp parallel for
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

                // adjust the row and column indicies for the current block, 
                // i.e., moving the origin to the top-left of this radar block.
                rdrY -= azimuthFirstLine;
                rdrX -= rangeFirstPixel;
                
                //store the adjusted X and Y indices 
                radarX[blockLine*_geoGridWidth + pixel] = rdrX;
                radarY[blockLine*_geoGridWidth + pixel] = rdrY;

            } // end loop over pixels of output grid 
        } // end loops over lines of output grid

        
        std::valarray<float> rtcDataBlock(0);
        if (_correctRtcFlag) {

            // resize the buffer for RTC
            rtcDataBlock.resize(rdrBlockLength * rdrBlockWidth);
        
            // get a block of RTC
            rtc.getBlock(rtcDataBlock,
                        rangeFirstPixel, azimuthFirstLine,
                        rdrBlockWidth, rdrBlockLength);

        }
        
        std::valarray<float> orientationAngleBlock(0);
        if (_correctOrientationFlag) {
            // buffer for the orientation angle
            orientationAngleBlock.resize(rdrBlockLength * rdrBlockWidth);
            // get a block of orientation angle
            orientationAngleRaster.getBlock(orientationAngleBlock,
                        rangeFirstPixel, azimuthFirstLine,
                        rdrBlockWidth, rdrBlockLength);

        }

        if (_dualPol) {

            //If dual-pol, correct RTC for each band and then  geocode it

            std::valarray<T> rdrDataBlock(rdrBlockLength * rdrBlockWidth);
            std::valarray<T> geoDataBlock(geoBlockLength * _geoGridWidth);


            //for each band in the input:
            for (size_t band = 0; band < nbands; ++band){

                std::cout << "band: " << band << std::endl;
                // get a block of data
                std::cout << "get data block " << std::endl;
                rdrCov.getBlock(rdrDataBlock,
                                rangeFirstPixel, azimuthFirstLine,
                                rdrBlockWidth, rdrBlockLength, band+1);

                if (_correctRtcFlag) {
                    // apply RTC correction factor
                    _correctRTC(rdrDataBlock, rtcDataBlock);
                }

                // Geocode; interpolate the data in radar grid to the geocoded grid
                std::cout << "interpolate " << std::endl;
                _interpolate(rdrDataBlock, geoDataBlock, radarX, radarY, 
                                rdrBlockWidth, rdrBlockLength, 
                                _geoGridWidth, geoBlockLength);

                // set output
                std::cout << "set output " << std::endl;
                geoCov.setBlock(geoDataBlock, 0, lineStart, 
                                _geoGridWidth, geoBlockLength, band+1);
            }
        } 
        else if (_quadPol) {
            
            std::cout << "needs to be implemented "  << std::endl;

        }


        // set output block of data
    } // end loop over block of output grid

    geoCov.setGeoTransform(_geoTrans);
    geoCov.setEPSG(_epsgOut);

}


template<class T>
void isce::signal::Covariance<T>::
_correctRTC(std::valarray<std::complex<float>> & rdrDataBlock, 
            std::valarray<float> & rtcDataBlock) {

    for (size_t i = 0; i<rdrDataBlock.size(); ++i)
        rdrDataBlock[i] = rdrDataBlock[i]*rtcDataBlock[i];

}

template<class T>
void isce::signal::Covariance<T>::
_correctRTC(std::valarray<std::complex<double>> & rdrDataBlock,
            std::valarray<float> & rtcDataBlock) {

    for (size_t i = 0; i<rdrDataBlock.size(); ++i)
        rdrDataBlock[i] = rdrDataBlock[i] * static_cast<double>(rtcDataBlock[i]);

}

template<class T>
void isce::signal::Covariance<T>::
faradayRotation(std::map<std::string, isce::io::Raster> & slc,  
                    isce::io::Raster & faradayAngleRaster,
                    size_t rangeLooks, size_t azimuthLooks)
{
    
    size_t numPolarizations = slc.size();
    std::cout << "number of polarizations : "<<  numPolarizations << std::endl;

    if (numPolarizations < 4) {
        // throw an error
        std::cout << "quad-pol data are required for Faraday rotation estimation" << std::endl; 
    }
    
    size_t nrows = slc["hh"].length();
    size_t ncols = slc["hh"].width();
    
    size_t blockRows = (_linesPerBlock/azimuthLooks)*azimuthLooks;
    size_t blockRowsMultiLooked = blockRows/azimuthLooks;
    size_t ncolsMultiLooked = ncols/rangeLooks;
    
    std::cout << blockRows << " , " << blockRowsMultiLooked << " , " << ncolsMultiLooked << std::endl;
    // number of blocks to process
    size_t nblocks = nrows / blockRows;
    if (nblocks == 0) {
        nblocks = 1;
    } else if (nrows % (nblocks * blockRows) != 0) {
        nblocks += 1;
    }
   
    std::cout << "number of blocks: " << nblocks << std::endl;

    // storage for a block of reference SLC data
    std::valarray<T> Shh(ncols*blockRows);
    std::valarray<T> Shv(ncols*blockRows);
    std::valarray<T> Svh(ncols*blockRows);
    std::valarray<T> Svv(ncols*blockRows);

    std::valarray<float> faradayAngle(ncolsMultiLooked*blockRowsMultiLooked);

    for (size_t block = 0; block < nblocks; ++block) {
        std::cout << "block: " << block << std::endl;       

        // start row for this block
        size_t rowStart;
        rowStart = block * blockRows;

        //number of lines of data in this block. blockRowsData<= blockRows
        //Note that blockRows is fixed number of lines
        //blockRowsData might be less than or equal to blockRows.
        //e.g. if nrows = 512, and blockRows = 100, then 
        //blockRowsData for last block will be 12
        size_t blockRowsData;
        if ((rowStart + blockRows) > nrows) {
            blockRowsData = nrows - rowStart;
        } else {
            blockRowsData = blockRows;
        }


        // get blocks of quad-pol data 
        slc["hh"].getBlock(Shh, 0, rowStart, ncols, blockRowsData);
        slc["hv"].getBlock(Shv, 0, rowStart, ncols, blockRowsData);
        slc["vh"].getBlock(Svh, 0, rowStart, ncols, blockRowsData);
        slc["vv"].getBlock(Svv, 0, rowStart, ncols, blockRowsData);

        // compute faraday rotation angle
        _faradayRotationAngle(Shh, Shv, Svh, Svv, faradayAngle, 
                                ncols, blockRows,
                                rangeLooks, azimuthLooks);

        faradayAngleRaster.setBlock(faradayAngle, 0, rowStart/azimuthLooks,
                                        ncols/rangeLooks, blockRowsData/azimuthLooks);

    }

}

template<class T>
void isce::signal::Covariance<T>::
_faradayRotationAngle(std::valarray<T>& Shh,
                    std::valarray<T>& Shv,
                    std::valarray<T>& Svh,
                    std::valarray<T>& Svv,
                    std::valarray<float>& faradayRotation,
                    size_t width, size_t length,
                    size_t rngLooks, size_t azLooks)
{

    size_t widthLooked = width/rngLooks;
    size_t lengthLooked = length/azLooks;

    size_t sizeData = Shh.size();
    std::valarray<float> M1(sizeData);
    std::valarray<float> M2(sizeData);
    std::valarray<float> M3(sizeData);

    for (size_t i = 0; i < sizeData; ++i ){
        M1[i] = -2*std::real((Shv[i] - Svh[i])*std::conj(Shh[i] + Svv[i]));
        M2[i] = std::pow(std::abs(Shh[i]+Svv[i]), 2.0);
        M3[i] = std::pow(std::abs(Shv[i]-Svh[i]), 2.0);
    }
   
    std::valarray<float> M1avg(widthLooked*lengthLooked);
    std::valarray<float> M2avg(widthLooked*lengthLooked);
    std::valarray<float> M3avg(widthLooked*lengthLooked);

    isce::signal::Looks<float> looksObj;
    looksObj.nrows(length);
    looksObj.ncols(width);
    looksObj.nrowsLooked(lengthLooked);
    looksObj.ncolsLooked(widthLooked);
    looksObj.rowsLooks(azLooks);
    looksObj.colsLooks(rngLooks);

    looksObj.multilook(M1, M1avg);
    looksObj.multilook(M2, M2avg);
    looksObj.multilook(M3, M3avg);
    
    size_t sizeOutput = faradayRotation.size();

    for (size_t i = 0; i < sizeOutput; ++i ){ 
        faradayRotation[i] = 0.25*std::atan2(M1[i], M2[i]-M3[i]);
    }
    
}

template<class T>
void isce::signal::Covariance<T>::
_correctFaradayRotation(isce::core::LUT2d<double>& faradayAngle, 
                        std::valarray<std::complex<float>>& Shh,
                    std::valarray<std::complex<float>>& Shv,
                    std::valarray<std::complex<float>>& Svh,
                    std::valarray<std::complex<float>>& Svv,
                    size_t length,
                    size_t width,
                    size_t lineStart)

{
    size_t sizeData = Shh.size();  

    for (size_t kk = 0; kk < length*width; ++kk) {
        size_t line = kk/width;
        size_t col = kk%width;
        size_t y = line + lineStart;
    
        double delta = faradayAngle.eval(y, col);
        float a = std::cos(delta);
        float b = std::sin(delta);

        T shh = a*a*Shh[kk] + a*b*Shv[kk] - a*b*Svh[kk] - b*b*Svv[kk];
        T shv = a*a*Shv[kk] + a*b*Shh[kk] - a*b*Svv[kk] + b*b*Svh[kk];
        T svh = a*b*Shh[kk] + b*b*Shv[kk] + a*a*Svh[kk] + a*b*Svv[kk];
        T svv = a*b*Shv[kk] - b*b*Shh[kk] + a*a*Svv[kk] - a*b*Svh[kk];

        Shh[kk] = shh;
        Shv[kk] = shv;
        Svh[kk] = svh;
        Svv[kk] = svv;

        
        
    }   
}


template<class T>
void isce::signal::Covariance<T>::
orientationAngle(isce::io::Raster& azimuthSlopeRaster,
                isce::io::Raster& rangeSlopeRaster,
                isce::io::Raster& lookAngleRaster,
                isce::io::Raster& tauRaster)
{
 
    
    size_t nrows = azimuthSlopeRaster.length();
    size_t ncols = azimuthSlopeRaster.width();
   
    size_t blockRows = _linesPerBlock;

    std::valarray<float> azimuthSlope(ncols*blockRows);
    std::valarray<float> rangeSlope(ncols*blockRows);
    std::valarray<float> lookAngle(ncols*blockRows);
    std::valarray<float> tau(ncols*blockRows);

    size_t nblocks = nrows / blockRows;
    if (nblocks == 0) {
        nblocks = 1;
    } else if (nrows % (nblocks * blockRows) != 0) {
        nblocks += 1;
    }

    for (size_t block = 0; block < nblocks; ++block) {
        std::cout << "block: " << block << std::endl;       

        // start row for this block
        size_t rowStart;
        rowStart = block * blockRows;

        //number of lines of data in this block. blockRowsData<= blockRows
        //Note that blockRows is fixed number of lines
        //blockRowsData might be less than or equal to blockRows.
        //e.g. if nrows = 512, and blockRows = 100, then 
        //blockRowsData for last block will be 12
        size_t blockRowsData;
        if ((rowStart + blockRows) > nrows) {
            blockRowsData = nrows - rowStart;
        } else {
            blockRowsData = blockRows;
        }

        azimuthSlopeRaster.getBlock(azimuthSlope, 0, rowStart, ncols, blockRowsData);
        rangeSlopeRaster.getBlock(rangeSlope, 0, rowStart, ncols, blockRowsData);
        lookAngleRaster.getBlock(lookAngle, 0, rowStart, ncols, blockRowsData);

        _orientationAngle(azimuthSlope, rangeSlope,
                        lookAngle, tau);

        tauRaster.setBlock(tau, 0, rowStart, ncols, blockRowsData);

    }
   
}


template<class T>
void isce::signal::Covariance<T>::
_orientationAngle(std::valarray<float>& azimuthSlope,
                std::valarray<float>& rangeSlope,
                std::valarray<float>& lookAngle,
                std::valarray<float>& tau)
{

    size_t sizeData = tau.size();
    for (size_t i = 0; i < sizeData; ++i ){
            tau = std::atan2(std::tan(azimuthSlope[i]), 
                    std::sin(lookAngle) - 
                        std::tan(rangeSlope)*std::cos(lookAngle));
    }
}


template<class T>
void isce::signal::Covariance<T>::
_correctOrientation(std::valarray<float>& tau, 
                    std::valarray<std::complex<float>>& C11,
                    std::valarray<std::complex<float>>& C12,
                    std::valarray<std::complex<float>>& C13,
                    std::valarray<std::complex<float>>& C21,
                    std::valarray<std::complex<float>>& C22,
                    std::valarray<std::complex<float>>& C23,
                    std::valarray<std::complex<float>>& C31,
                    std::valarray<std::complex<float>>& C32,
                    std::valarray<std::complex<float>>& C33)

{
    // Given the 3x3 Covariance matrix, the matrix after 
    // polarimetric orientation correction is obtained as:
    // C = R*C*R_T
    // where R is the rotation matrix and R_T is the transpose 
    // of the rotation matrix.
    
    // the size of the rotation angle array
    size_t arraySize = tau.size();

    // buffer for the first two elements of the rotation matrix
    std::valarray<float> R11(arraySize);
    std::valarray<float> R12(arraySize);
    
    // the first two elements of the rotation matrix
    R11 = 1.0 + std::cos(2.0f*tau);
    R12 = std::sqrt(2.0)*std::sin(2.0f*tau);
    
    // All other elemensts of the rotation matrix
    // can be derived from the first two elements. 
    // R13 = 2.0 - R11;
    // R21 = -1.0*R12;
    // R22 = 2.0*(R11 - 1.0);
    // R23 = R12;
    // R31 = 2.0 - R11;
    // R32 = -1.0*R12; 
    // R33 = R11;
    // Therefore there is no need to compute them 


    for (size_t i = 0; i < arraySize; ++i) {
        float r11 = R11[i];
        float r12 = R12[i];
        float r13 = (2.0f - R11[i]);
        float r21 = -1.0f*R12[i];
        float r22 = 2.0f*(R11[i] - 1.0f);
        float r23 = R12[i];
        float r31 = 2.0f - R11[i];
        float r32 = -1.0f*R12[i];
        float r33 = R11[i];
       
        std::complex<float> c11 = 0.25f*(r11*(C11[i]*r11 + C12[i]*r12 + C13[i]*r13) +
                        r12*(C21[i]*r11 + C22[i]*r12 + C23[i]*r13) +
                        r13*(C31[i]*r11 + C32[i]*r12 + C33[i]*r13));

        std::complex<float> c12 = 0.25f*(r11*(C11[i]*r21 + C12[i]*r22 + C13[i]*r23) +
                        r12*(C21[i]*r21 + C22[i]*r22 + C23[i]*r23) +
                        r13*(C31[i]*r21 + C32[i]*r22 + C33[i]*r23));

        std::complex<float> c13 = 0.25f*(r11*(C11[i]*r31 + C12[i]*r32 + C13[i]*r33) +
                        r12*(C21[i]*r31 + C22[i]*r32 + C23[i]*r33) +
                        r13*(C31[i]*r31 + C32[i]*r32 + C33[i]*r33));

        std::complex<float> c21 = 0.25f*(r21*(C11[i]*r11 + C12[i]*r12 + C13[i]*r13) +
                        r22*(C21[i]*r11 + C22[i]*r12 + C23[i]*r13) +
                        r23*(C31[i]*r11 + C32[i]*r12 + C33[i]*r13));

        std::complex<float> c22 = 0.25f*(r21*(C11[i]*r21 + C12[i]*r22 + C13[i]*r23) +
                        r22*(C21[i]*r21 + C22[i]*r22 + C23[i]*r23) +
                        r23*(C31[i]*r21 + C32[i]*r22 + C33[i]*r23));

        std::complex<float> c23 = 0.25f*(r21*(C11[i]*r31 + C12[i]*r32 + C13[i]*r33) +
                        r22*(C21[i]*r31 + C22[i]*r32 + C23[i]*r33) +
                        r23*(C31[i]*r31 + C32[i]*r32 + C33[i]*r33));

        std::complex<float> c31 = 0.25f*(r31*(C11[i]*r11 + C12[i]*r12 + C13[i]*r13) +
                        r32*(C21[i]*r11 + C22[i]*r12 + C23[i]*r13) +
                        r33*(C31[i]*r11 + C32[i]*r12 + C33[i]*r13));

        std::complex<float> c32 = 0.25f*(r31*(C11[i]*r21 + C12[i]*r22 + C13[i]*r23) +
                        r32*(C21[i]*r21 + C22[i]*r22 + C23[i]*r23) +
                        r33*(C31[i]*r21 + C32[i]*r22 + C33[i]*r23));

        std::complex<float> c33 = 0.25f*(r31*(C11[i]*r31 + C12[i]*r32 + C13[i]*r33) +
                        r32*(C21[i]*r31 + C22[i]*r32 + C23[i]*r33) +
                        r33*(C31[i]*r31 + C32[i]*r32 + C33[i]*r33));

        C11[i] = c11;
        C12[i] = c12;
        C13[i] = c13;
        C21[i] = c21;
        C22[i] = c22;
        C23[i] = c23;
        C31[i] = c31;
        C32[i] = c32;
        C33[i] = c33;


    }
}

/*

void isce::signal::Covariance::
_correctFaradayRotation()
{
    
}

void isce::signal::Covariance::
_symmetrization()
{
    
}

*/


template<class T>
void isce::signal::Covariance<T>::
_interpolate(std::valarray<T>& rdrDataBlock,
            std::valarray<T>& geoDataBlock,
            std::valarray<double>& radarX, std::valarray<double>& radarY,
            size_t radarBlockWidth, size_t radarBlockLength,
            size_t width, size_t length)
{

    double extraMargin = 4.0;
    for (size_t i = 0; i< length; ++i) {
        for (size_t j = 0; j < width; ++j) {

            // if this point falls somewhere within the radar data box,
            // then perform the interpolation

            if (radarX[i*width + j] >= extraMargin &&
                    radarY[i*width + j] >= extraMargin &&
                    radarX[i*width + j] < (radarBlockWidth - extraMargin) &&
                    radarY[i*width + j] < (radarBlockLength - extraMargin) ) {

                geoDataBlock[i*width + j] = _interp->interpolate(radarX[i*width + j],
                                                radarY[i*width + j], rdrDataBlock, radarBlockWidth);

            }
        }
    }

}

/*
template<class T>
void isce::signal::Covariance<T>::
_interpolate(isce::core::Matrix<T>& rdrDataBlock, 
            isce::core::Matrix<T>& geoDataBlock,
            std::valarray<double>& radarX, std::valarray<double>& radarY, 
            int radarBlockWidth, int radarBlockLength)
{

    size_t length = geoDataBlock.length();
    size_t width = geoDataBlock.width();
    double extraMargin = 4.0;
    for (size_t i = 0; i< length; ++i) {
        for (size_t j = 0; j < width; ++j) {

            // if this point falls somewhere within the radar data box, 
            // then perform the interpolation

            if (radarX[i*width + j] >= extraMargin &&
                    radarY[i*width + j] >= extraMargin &&
                    radarX[i*width + j] < (radarBlockWidth - extraMargin) &&
                    radarY[i*width + j] < (radarBlockLength - extraMargin) ) {

                geoDataBlock(i,j) = _interp->interpolate(radarX[i*width + j], 
                                                radarY[i*width + j], rdrDataBlock);
            
            }
        }
    }

}
*/


template<class T>
void isce::signal::Covariance<T>::
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

    if (demInterp.width() == 0 || demInterp.length() == 0)
        std::cout << "warning there are not enough DEM coverage in the bounding box. " << std::endl;

    // declare the dem interpolator
    demInterp.declare();
}

template<class T>
void isce::signal::Covariance<T>::
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

    // compute geo2rdr for the 4 corners
    for (size_t i = 0; i<4; ++i){
        _geo2rdr(X[i], Y[i], azimuthTime[i], slantRange[i], demInterp); 
    }

    // the first azimuth line
    azimuthFirstLine = (azimuthTime.min() - 
                            _azimuthStartTime.secondsSinceEpoch(_refEpoch))/
                                _azimuthTimeInterval;

    // the last azimuth line
    azimuthLastLine = (azimuthTime.max() - 
                            _azimuthStartTime.secondsSinceEpoch(_refEpoch))/
                                _azimuthTimeInterval;

    // the first and last range pixels 
    rangeFirstPixel = (slantRange.min() - _startingRange)/_rangeSpacing;
    rangeLastPixel = (slantRange.max() - _startingRange)/_rangeSpacing;

    // extending the radar bounding box by the extra margin
    azimuthFirstLine -= margin;
    azimuthLastLine  += margin;
    rangeFirstPixel -= margin;
    rangeLastPixel  += margin;

    // make sure the radar block's bounding box is inside the existing radar grid
    if (azimuthFirstLine < 0)
        azimuthFirstLine = 0;

    if (azimuthLastLine > (_radarGridLength - 1))
        azimuthLastLine = _radarGridLength - 1;

    if (rangeFirstPixel < 0)
        rangeFirstPixel = 0;

    if (rangeLastPixel > (_radarGridWidth - 1))
        rangeLastPixel = _radarGridWidth - 1;

}

template<class T>
void isce::signal::Covariance<T>::
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
                    llh, _ellipsoid, _orbit, _doppler, _mode, 
                    azimuthTime, slantRange, _threshold,
                    _numiter, 1.0e-8);

}

template class isce::signal::Covariance<std::complex<float>>;
template class isce::signal::Covariance<std::complex<double>>;


