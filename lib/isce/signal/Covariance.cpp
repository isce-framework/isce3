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

    bool dualPol = false;
    bool quadPol = false;

    std::string coPol;
    std::string crossPol;

    if (numPolarizations == 1){
        std::cout << "Covariance estimation needs at least two polarizations" << std::endl;
    } else if (numPolarizations == 2 && numCovElements == 3) {

        dualPol = true;

        if (slc.count("hh") > 0)
            coPol = "hh";
        else if (slc.count("vv") > 0)
            coPol = "vv";

        if (slc.count("hv") > 0)
            crossPol = "hv";
        else if (slc.count("vh") > 0)
            crossPol = "vh";

    } else if (numPolarizations == 4 && numCovElements == 10) {
        quadPol = true;
    } 

    // instantiate the crossmul object
    isce::signal::Crossmul crsmul;

    // set up crossmul
    /*crsmul.doppler(_doppler, _doppler);

    crsmul.prf(_prf);

    crsmul.rangeSamplingFrequency(_rangeSamplingFrequency);

    crsmul.rangeBandwidth(_rangeBandwidth);

    crsmul.wavelength(_wavelength);

    crsmul.rangePixelSpacing(_rangePixelSpacing);
    */
    crsmul.rangeLooks(_rangeLooks);

    crsmul.azimuthLooks(_azimuthLooks);

    crsmul.doCommonAzimuthbandFiltering(false);

    crsmul.doCommonRangebandFiltering(false);   

    if (dualPol) {

        crsmul.crossmul(slc[coPol], slc[coPol], cov[std::make_pair(coPol, coPol)]);

        crsmul.crossmul(slc[coPol], slc[crossPol], cov[std::make_pair(coPol, crossPol)]);

        crsmul.crossmul(slc[crossPol], slc[crossPol], cov[std::make_pair(crossPol, crossPol)]);


    } else if (quadPol){

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
                isce::io::Raster& rtc,
                isce::io::Raster & demRaster,
                isce::io::Raster& geoCov) {

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

        // define the matrix based on the rasterbands data type

        std::valarray<T> rdrDataBlock(rdrBlockLength * rdrBlockWidth);
        std::valarray<float> rtcDataBlock(rdrBlockLength * rdrBlockWidth);
        std::valarray<T> geoDataBlock(geoBlockLength * _geoGridWidth);

         
        //for each band in the input:
        for (size_t band = 0; band < nbands; ++band){

            std::cout << "band: " << band << std::endl;
            // get a block of data
            std::cout << "get data block " << std::endl;
            rdrCov.getBlock(rdrDataBlock,
                                rangeFirstPixel, azimuthFirstLine,
                                rdrBlockWidth, rdrBlockLength, band+1);

            rtc.getBlock(rtcDataBlock,
                                rangeFirstPixel, azimuthFirstLine,
                                rdrBlockWidth, rdrBlockLength, band+1);

            // apply RTC correction factor
            //rdrDataBlock *= rtcDataBlock;
            _correctRTC(rdrDataBlock, rtcDataBlock);

            // interpolate the data in radar grid to the geocoded grid
            std::cout << "interpolate " << std::endl;
            _interpolate(rdrDataBlock, geoDataBlock, radarX, radarY, 
                                rdrBlockWidth, rdrBlockLength, 
                                _geoGridWidth, geoBlockLength);

            // set output
            std::cout << "set output " << std::endl;
            geoCov.setBlock(geoDataBlock, 0, lineStart, 
                                _geoGridWidth, geoBlockLength, band+1);
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
    _applyFaradayRotation = false;
    isce::io::Raster dummySlc("/vsimem/dummy", 1, 1, 1, GDT_CFloat32, "VRT");
   
    std::map<std::string, isce::io::Raster> correctedSlc =
                {{"hh", dummySlc}};

    faradayRotation(slc, faradayAngleRaster, correctedSlc,
                    rangeLooks, azimuthLooks);

}

template<class T>
void isce::signal::Covariance<T>::
faradayRotation(std::map<std::string, isce::io::Raster> & slc,  
                    isce::io::Raster & faradayAngleRaster,
                    std::map<std::string, isce::io::Raster> & correctedSlc,
                    size_t rangeLooks, size_t azimuthLooks)
{
    
    size_t numPolarizations = slc.size();
    if (numPolarizations < 4) {
        // throw an error
        std::cout << "quad-pol data are required for Faraday rotation estimation" << std::endl; 
    }
    
    size_t nrows = slc["hh"].length();
    size_t ncols = slc["hh"].width();

    size_t blockRows = (blockRows/azimuthLooks)*azimuthLooks;
    size_t blockRowsMultiLooked = blockRows/azimuthLooks;
    size_t ncolsMultiLooked = ncols/rangeLooks;

    // number of blocks to process
    size_t nblocks = nrows / blockRows;
    if (nblocks == 0) {
        nblocks = 1;
    } else if (nrows % (nblocks * blockRows) != 0) {
        nblocks += 1;
    }

    // storage for a block of reference SLC data
    std::valarray<T> Shh(nrows*blockRows);
    std::valarray<T> Shv(nrows*blockRows);
    std::valarray<T> Svh(nrows*blockRows);
    std::valarray<T> Svv(nrows*blockRows);

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

        // apply faraday rotation angle
        if (_applyFaradayRotation) {
            std::cout << "apply Faraday rotation angle" << std::endl;
        }
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
        M2[i] = std::pow(std::abs(Shh[i]-Svv[i]), 2.0);
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
        faradayRotation[i] = -0.25*std::atan2(M1[i], M2[i]-M3[i]);
    }
    
}

/*
void isce::signal::Covariance::
geocodeCovariance(isce::io::Raster& C11,
                isce::io::Raster& C12,
                isce::io::Raster& C22,
                isce::io::Raster& TCF,
                isce::io::Raster& GC11,
                isce::io::Raster& GC12,
                isce::io::Raster& GC13,
                isce::io::Raster& GC21,
                isce::io::Raster& GC22,
                isce::io::Raster& GC23,
                isce::io::Raster& GC31,
                isce::io::Raster& GC32,
                isce::io::Raster& GC33)
{
    
    // buffers for blocks of data

    //for each block in the geocoded grid:
        
        //read a block in radar coordintae for C11, C12, C22, RTC
        
        //RTC correction
        
        //Polarization estimation/correction
        
        //Effective number of looks
        
        //Faraday rotation estimation/correction
        
        //Symmetrization
                
        //Covariance 
    
}

void isce::signal::Covariance::
_rtcCorrection(std::valarray<std::complex<float>>& input, 
                std::valarray<float>& TCF)
{
    input *= TCF; 
}

void isce::signal::Covariance::
_orientationAngle(std::valarray<float>& azimuthSlope,
                std::valarray<float>& rangeSlope,
                std::valarray<float>& lookAngle,
                std::valarray<float>& tau)
{
    tau = std::atan2(std::tan(azimuthSlope), 
                    std::sin(lookAngle) - 
                        std::tan(rangeSlope)*std::cos(lookAngle));

}

void isce::signal::Covariance::
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
    size_t arraySize = tau.size();
    std::valarray<float> R11(arraySize);
    std::valarray<float> R12(arraySize);
    


    R11 = 1.0 + std::cos(2*tau);
    R12 = std::sqrt(2)*std::sin(2*tau);
    R13 = 2.0 - R11;

    R21 = -1.0*R12;
    R22 = 2.0*(R11 - 1.0);
    R23 = R12;

    R31 = 2.0 - R11;
    R32 = -1*R12; 
    R33 = R11;

    // 
    c11 = 0.25*(R11*(C11*R11 + C12*R12 + C13*R13) +
                R12*(C21*R11 + C22*R12 + C23*R13) +
                R13*(C31*R11 + C32*R12 + C33*R13));

    c12 = 0.25*(R11*(C11*R21 + C12*R22 + C13*R23) + 
                R12*(C21*R21 + C22*R22 + C23*R23) + 
                R13*(C31*R21 + C32*R22 + C33*R23));

    c13 = 0.25*(R11*(C11*R31 + C12*R32 + C13*R33) +
                R12*(C21*R31 + C22*R32 + C23*R33) +
                R13*(C31*R31 + C32*R32 + C33*R33));

    c21 = 0.25*(R21*(C11*R11 + C12*R12 + C13*R13) +
                R22*(C21*R11 + C22*R12 + C23*R13) +
                R23*(C31*R11 + C32*R12 + C33*R13));

    c22 = 0.25*(R21*(C11*R21 + C12*R22 + C13*R23) +
                R22*(C21*R21 + C22*R22 + C23*R23) +
                R23*(C31*R21 + C32*R22 + C33*R23));

    c23 = 0.25*(R21*(C11*R31 + C12*R32 + C13*R33) +
                R22*(C21*R31 + C22*R32 + C23*R33) +
                R23*(C31*R31 + C32*R32 + C33*R33));

    c31 = 0.25*(R31*(C11*R11 + C12*R12 + C13*R13) +
                R32*(C21*R11 + C22*R12 + C23*R13) +
                R33*(C31*R11 + C32*R12 + C33*R13));

    c32 = 0.25*(R31*(C11*R21 + C12*R22 + C13*R23) +
                R32*(C21*R21 + C22*R22 + C23*R23) +
                R33*(C31*R21 + C32*R22 + C33*R23));

    c33 = 0.25*(R31*(C11*R31 + C12*R32 + C13*R33) +
                R32*(C21*R31 + C22*R32 + C23*R33) +
                R33*(C31*R31 + C32*R32 + C33*R33));

}

void isce::signal::Covariance::
_faradayRotationAngle(std::valarray<std::complex<float>>& Shh,
                    std::valarray<std::complex<float>>& Shv,
                    std::valarray<std::complex<float>>& Svh,
                    std::valarray<std::complex<float>>& Svv,
                    std::valarray<float>& delta)
{
    delta = 0.25*std::atan2(-2*std::real((Shv - Svh)*std::conj(Shh + Svv)) ,
                            std::pow(std::abs(Shh + Svv), 2) - 
                            std::pow(std::abs(Shv - Svh), 2) );


}

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


