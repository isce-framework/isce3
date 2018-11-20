// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi, Bryan Riel
// Copyright 2018-
//

#include "Crossmul.h"

/*
isce::signal::Crossmul::
Crossmul(const isce::product::Product& referenceSlcProduct,
         const isce::product::Product& secondarySlcProduct,
         isce::product::Product& outputInterferogramProduct)
*/


/**
* @param[in] referenceSLC Raster object of refernce SLC
* @param[in] secondarySLC Raster object of secondary SLC
* @param[out] interferogram Raster object of output interferogram
*/


void isce::signal::Crossmul::
crossmul(isce::io::Raster& referenceSLC,
        isce::io::Raster& secondarySLC,
        isce::io::Raster& interferogram)
{

    _doCommonRangebandFilter = false;
    isce::io::Raster rngOffsetRaster("/vsimem/dummy", 1, 1, 1, GDT_CFloat32, "ENVI");
    crossmul(referenceSLC, 
            secondarySLC, 
            interferogram, 
            rngOffsetRaster);

}

void isce::signal::Crossmul::
crossmul(isce::io::Raster& referenceSLC,
        isce::io::Raster& secondarySLC,
        isce::io::Raster& interferogram,
        isce::io::Raster& rngOffsetRaster)
{

    // Create reusable pyre::journal channels
    pyre::journal::warning_t warning("isce.geometry.Topo");
    pyre::journal::info_t info("isce.geometry.Topo");

    size_t nrows = referenceSLC.length();
    size_t ncols = referenceSLC.width();

    //signal object for refSlc
    isce::signal::Signal<float> refSignal;

    //signal object for secSlc
    isce::signal::Signal<float> secSignal;

    // Compute FFT size (power of 2)
    size_t nfft;
    refSignal.nextPowerOfTwo(ncols, nfft);

    // number of blocks to process
    size_t nblocks = nrows / blockRows;
    if (nblocks == 0) {
        nblocks = 1;
    } else if (nrows % (nblocks * blockRows) != 0) {
        nblocks += 1;
    }

    // storage for a block of reference SLC data
    std::valarray<std::complex<float>> refSlc(nfft*blockRows);

    // storage for a block of secondary SLC data
    std::valarray<std::complex<float>> secSlc(nfft*blockRows);

    // storage for a block of range offsets
    std::valarray<double> rngOffset(nfft*blockRows);

    // storage for spectrum of the block of data in reference SLC
    std::valarray<std::complex<float>> refSpectrum(nfft*blockRows);

    // storage for spectrum of the block of data in secondary SLC
    std::valarray<std::complex<float>> secSpectrum(nfft*blockRows);

    // upsampled spectrum of the block of reference SLC
    std::valarray<std::complex<float>> refSpectrumUpsampled(oversample*nfft*blockRows);

    // upsampled spectrum of the block of secondary SLC
    std::valarray<std::complex<float>> secSpectrumUpsampled(oversample*nfft*blockRows);

    // upsampled block of reference SLC 
    std::valarray<std::complex<float>> refSlcUpsampled(oversample*nfft*blockRows);

    // upsampled block of secondary SLC
    std::valarray<std::complex<float>> secSlcUpsampled(oversample*nfft*blockRows);

    // upsampled interferogram
    std::valarray<std::complex<float>> ifgramUpsampled(oversample*ncols*blockRows);

    // full resolution interferogram
    std::valarray<std::complex<float>> ifgram(ncols*blockRows);

    // make forward and inverse fft plans for the reference SLC 
    refSignal.forwardRangeFFT(refSlc, refSpectrum, nfft, blockRows);
    refSignal.inverseRangeFFT(refSpectrumUpsampled, refSlcUpsampled, nfft*oversample, blockRows);

    // make forward and inverse fft plans for the secondary SLC
    secSignal.forwardRangeFFT(secSlc, secSpectrum, nfft, blockRows);
    secSignal.inverseRangeFFT(secSpectrumUpsampled, secSlcUpsampled, nfft*oversample, blockRows);

    // looking down the upsampled interferogram may shift the samples by a fraction of a pixel
    // depending on the oversample factor. predicting the impact of the shift in frequency domain 
    // which is a linear phase allows to account for it during the upsampling process
    std::valarray<std::complex<float>> shiftImpact(oversample*nfft*blockRows);
    lookdownShiftImpact(oversample,  nfft, 
                        blockRows, shiftImpact);

    //filter objects which will be used for azimuth and range common band filtering
    isce::signal::Filter<float> azimuthFilter;
    isce::signal::Filter<float> rangeFilter;

    rangeFilter.initiateRangeFilter(refSlc, refSpectrum, nfft, blockRows);

    // storage for azimuth spectrum used by filter
    std::valarray<std::complex<float>> refAzimuthSpectrum(nfft*blockRows);

    if (_doCommonAzimuthbandFilter){
        // construct azimuth common band filter for a block of data
        azimuthFilter.constructAzimuthCommonbandFilter(_refDoppler, 
					    _secDoppler, 
                                            _commonAzimuthBandwidth,
                                            _prf, 
                                            _beta,
                                            refSlc, refAzimuthSpectrum,
                                            nfft, blockRows);
    }

    // loop over all blocks
    std::cout << "nblocks : " << nblocks << std::endl;

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
        if ((rowStart+blockRows)>nrows) {
            blockRowsData = nrows - rowStart;
        } else {
            blockRowsData = blockRows;
        }

        // fill the valarray with zero before getting the block of the data
        refSlc = 0;
        secSlc = 0;
        ifgramUpsampled = 0;
        ifgram = 0;

        // get a block of reference and secondary SLC data
        // This will change once we have the functionality to 
        // get a block of data directly in to a slice
        std::valarray<std::complex<float>> dataLine(ncols);
        for (size_t line = 0; line < blockRowsData; ++line){
            referenceSLC.getLine(dataLine, rowStart + line);
            refSlc[std::slice(line*nfft, ncols, 1)] = dataLine;

            secondarySLC.getLine(dataLine, rowStart + line);
            secSlc[std::slice(line*nfft, ncols, 1)] = dataLine;
        }
        //referenceSLC.getBlock(refSlc, 0, rowStart, ncols, blockRowsData);
        //secondarySLC.getBlock(secSlc, 0, rowStart, ncols, blockRowsData);
    
        //commaon azimuth band-pass filter the reference and secondary SLCs
        if (_doCommonAzimuthbandFilter){
            std::cout << "filter the refSlc " << std::endl;
            azimuthFilter.filter(refSlc, refAzimuthSpectrum);
            std::cout << "filter the secSlc " << std::endl;
            azimuthFilter.filter(secSlc, refAzimuthSpectrum);
        }

        // common range band-pass filtering
        if (_doCommonRangebandFilter){

            // get a block of range offsets
            std::valarray<double> offsetLine(ncols);
            for (size_t line = 0; line < blockRowsData; ++line){
                rngOffsetRaster.getLine(offsetLine, rowStart + line);
                rngOffset[std::slice(line*nfft, ncols, 1)] = offsetLine;
            }

            // do the range common band filter
            rangeCommonBandFilter(refSlc,
                                secSlc,
                                rngOffset,
                                _rangePixelSpacing,
                                _wavelength,
                                blockRows,
                                nfft,
                                rangeFilter,
                                refSpectrum);
        }

        // upsample the refernce and secondary SLCs
        refSignal.upsample(refSlc, refSlcUpsampled, blockRows, nfft, oversample, shiftImpact);
        secSignal.upsample(secSlc, secSlcUpsampled, blockRows, nfft, oversample, shiftImpact);
       
        // Compute oversampled interferogram data
        for (size_t line = 0; line < blockRowsData; line++){
            for (size_t col = 0; col < oversample*ncols; col++){
                ifgramUpsampled[line*(oversample*ncols) + col] = 
                        refSlcUpsampled[line*(oversample*nfft) + col]*
                        std::conj(secSlcUpsampled[line*(oversample*nfft) + col]);
            }
        }

        // Reclaim the extra oversample looks across
        for (size_t line = 0; line < blockRowsData; line++){
            for (size_t col = 0; col < ncols; col++){
                std::complex<float> sum =(0,0);
                for (size_t j=0; j< oversample; j++)
                    sum += ifgramUpsampled[line*(ncols*oversample) + j + col*oversample];
                ifgram[line*ncols + col] = sum;            
            }
        }

	// Take looks down (summing columns)
        
	// set the block of interferogram
        interferogram.setBlock(ifgram, 0, rowStart, ncols, blockRowsData);

    }
}


void isce::signal::Crossmul::
lookdownShiftImpact(size_t oversample, size_t nfft, size_t blockRows, 
		std::valarray<std::complex<float>> &shiftImpact)
{
    // range frequencies given nfft and oversampling factor
    std::valarray<double> rangeFrequencies(oversample*nfft);

    // sampling in range
    double dt = 1.0;///_rangeSamplingFrequency;

    // get the vector of range frequencies
    //filter object
    isce::signal::Filter<float> tempFilter;
    tempFilter.fftfreq(oversample*nfft, dt, rangeFrequencies);

    // in the process of upsampling the SLCs, creating upsampled interferogram
    // and then looking down the upsampled interferogram to the original size of
    // the SLCs, a shift is introduced in range direction.
    // As an example for a signal with length of 5 and :
    // original sample locations:   0       1       2       3        4
    // upsampled sample locations:  0   0.5 1  1.5  2  2.5  3   3.5  4   4.5
    // Looked dow sample locations:   0.25    1.25    2.25    3.25    4.25
    // Obviously the signal after looking down would be shifted by 0.25 pixel in
    // range comared to the original signal. Since a shift in time domain introcues
    // a liner phase in frequency domain, here is the impact in frequency domain

    // the constant shift based on the oversampling factor
    double shift = 0.0;
    if (oversample == 2){
        shift = 0.25;
    } else if (oversample > 2){
        shift = (1.0 - 1.0/oversample)/2.0;
    }

    std::valarray<std::complex<float>> shiftImpactLine(oversample*nfft);
    for (size_t col=0; col<shiftImpactLine.size(); ++col){
        double phase = -1.0*shift*2.0*M_PI*rangeFrequencies[col];
        shiftImpactLine[col] = std::complex<float> (std::cos(phase),
                                                    std::sin(phase));
    }

    // The imapct is the same for each range line. Therefore copying the line for the block
    for (size_t line = 0; line < blockRows; ++line){
            shiftImpact[std::slice(line*nfft*oversample, nfft*oversample, 1)] = shiftImpactLine;
    }
}

void isce::signal::Crossmul::
rangeCommonBandFilter(std::valarray<std::complex<float>> &refSlc,
                        std::valarray<std::complex<float>> &secSlc,
                        std::valarray<double> rngOffset,
                        double rangePixelSpacing,
                        double wavelength,
                        size_t blockLength,
                        size_t ncols)
{
    // size of the arrays 
    size_t vectorLength = refSlc.size();

    std::valarray<std::complex<float>> tempRefSlc(vectorLength);
    std::valarray<std::complex<float>> tempSecSlc(vectorLength);

    // Shifting the range spectrum of each image according to the local (slope-dependent) wavenumber.
    // This shift in frequency domain is achieved by removing/adding the geometrical (representing topography) 
    // from/to refernce and secondary SLCs in time domain. 
    for (size_t i = 0; i < vectorLength; ++i){

        // the phase due to baseline separation obtained from range difference 
        // from refernce and secondary antennas to the target (i.e., range offser derived from 
        // geometrical coregistration)
        double phase = 4.0*M_PI*rangePixelSpacing*rngOffset[i]/wavelength;

        // refSLc = refSlc*exp(-1J*phase)
        tempRefSlc[i] = refSlc[i] * std::complex<float>(std::cos(phase), -1.0*std::sin(phase));

        // refSLc = secSlc*exp(1J*phase)
        tempSecSlc[i] = secSlc[i] * std::complex<float> (std::cos(phase), std::sin(phase));

    }

    // low pass filter the ref and sec slc
    // For now we low-pass filter the reference and secondary SLCs with a simple
    // hamming window
    float hwCoef1 = 0.23;
    float hwCoef2 = 0.54;
    float hwCoef3 = 0.23;
    for (size_t line = 0; line < blockLength; ++line){
        for (size_t col = 1; col < ncols-1; ++col){

            refSlc[line*ncols + col] = hwCoef1*tempRefSlc[line*ncols + col - 1] +
                                            hwCoef2*tempRefSlc[line*ncols + col] +
                                            hwCoef3*tempRefSlc[line*ncols + col + 1];

            secSlc[line*ncols + col] = hwCoef1*tempSecSlc[line*ncols + col-1] + 
                                            hwCoef2*tempSecSlc[line*ncols + col] + 
                                            hwCoef3*tempSecSlc[line*ncols + col + 1];
        }
    }
    
    
    // add/remove half geometrical phase to/from reference and secondary SLCs
    for (size_t i = 0; i < vectorLength; ++i){

        // Half phase due to baseline separation obtained from range difference
        // from refernce and secondary antennas to the target (i.e., range offser derived from
        // geometrical coregistration)
        double halfPhase = 2.0*M_PI*rangePixelSpacing*rngOffset[i]/wavelength;

        // refSLc = refSlc*exp(1J*halfPhase)
        refSlc[i] = refSlc[i] * std::complex<float> (std::cos(halfPhase), std::sin(halfPhase));

        // refSLc = secSlc*exp(-1J*halfPhase)
        secSlc[i] = secSlc[i] * std::complex<float> (std::cos(halfPhase), -1*std::sin(halfPhase));

    }


}

void isce::signal::Crossmul::
rangeCommonBandFilter(std::valarray<std::complex<float>> &refSlc,
                        std::valarray<std::complex<float>> &secSlc,
                        std::valarray<double> rngOffset,
                        double rangePixelSpacing,
                        double wavelength,
                        size_t blockLength,
                        size_t ncols,
                        isce::signal::Filter<float> &rngFilter,
                        std::valarray<std::complex<float>> &spectrum)
{
    // size of the arrays
    size_t vectorLength = refSlc.size();

    // Shifting the range spectrum of each image according to the local 
    // (slope-dependent) wavenumber. This shift in frequency domain is 
    // achieved by removing/adding the geometrical (representing topography)
    // from/to refernce and secondary SLCs in time domain.
    for (size_t i = 0; i < vectorLength; ++i){

        // the phase due to baseline separation obtained from range difference
        // from refernce and secondary antennas to the target (i.e., range offser derived from
        // geometrical coregistration)
        double phase = 4.0*M_PI*rangePixelSpacing*rngOffset[i]/wavelength;

        // refSLc = refSlc*exp(-1J*phase)
        refSlc[i] = refSlc[i] * std::complex<float> (std::cos(phase), -1.0*std::sin(phase));

        // refSLc = secSlc*exp(1J*phase)
        secSlc[i] = secSlc[i] * std::complex<float> (std::cos(phase), std::sin(phase));

    }

    // determine the frequency shift for now keep it zero
    double frequencyShift = 0.0;

    // low pass filter the ref and sec slc
    std::valarray<double> filterCenterFrequency{0.0};
    std::valarray<double> filterBandwidth{_rangeBandwidth - frequencyShift};
    std::string filterType = "cosine";
    rngFilter.constructRangeBandpassFilter(_rangeSamplingFrequency,
                                    filterCenterFrequency,
                                    filterBandwidth,
                                    ncols,
                                    blockLength,
                                    filterType);

        
    rngFilter.filter(refSlc, spectrum);
    rngFilter.filter(secSlc, spectrum);

    // add/remove half geometrical phase to/from reference and secondary SLCs
    for (size_t i = 0; i < vectorLength; ++i){

        // Half phase due to baseline separation obtained from range difference
        // from refernce and secondary antennas to the target (i.e., range offser derived from
        // geometrical coregistration)
        double halfPhase = 2.0*M_PI*rangePixelSpacing*rngOffset[i]/wavelength;

        // refSLc = refSlc*exp(1J*halfPhase)
        refSlc[i] = refSlc[i] * std::complex<float> (std::cos(halfPhase), std::sin(halfPhase));

        // refSLc = secSlc*exp(-1J*halfPhase)
        secSlc[i] = secSlc[i] * std::complex<float> (std::cos(halfPhase), -1*std::sin(halfPhase));

    }


}


