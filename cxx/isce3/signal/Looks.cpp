// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2018-
//

#include "Looks.h"

bool isce3::signal::verifyComplexToRealCasting(isce3::io::Raster& input_raster,
                                              isce3::io::Raster& output_raster,
                                              int& exponent) {
    GDALDataType input_dtype = input_raster.dtype();
    GDALDataType output_dtype = output_raster.dtype();
    bool flag_complex_to_real = (GDALDataTypeIsComplex(input_dtype) &&
                                 !GDALDataTypeIsComplex(output_dtype));
    if (exponent == 0 && flag_complex_to_real)
        exponent = 2;
    else if (exponent == 0)
        exponent = 1;
    else if (exponent != 1 && !flag_complex_to_real) {
        std::string error_message =
                "ERROR multilooking with non-unitary exponents";
        error_message += " is only implemented for complex inputs (SLCs) with "
                         "real outputs";
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_message);
    }
    return flag_complex_to_real;
}

template<class T>
void isce3::signal::Looks<T>::multilook(isce3::io::Raster& input_raster,
                                       isce3::io::Raster& output_raster,
                                       int exponent) {
    int nbands = input_raster.numBands();
    _ncols = input_raster.width();
    _nrows = input_raster.length();
    _ncolsLooked = _ncols / _colsLooks;
    _nrowsLooked = _nrows / _rowsLooks;

    bool flag_complex_to_real =
            verifyComplexToRealCasting(input_raster, output_raster, exponent);
    for (int band = 0; band < nbands; band++) {
        if (nbands == 1)
            std::cout << "loading slant-range image..." << std::endl;
        else
            std::cout << "loading slant-range band: " << band << std::endl;
        std::valarray<T> image_ml(_ncolsLooked * _nrowsLooked);
        if (!flag_complex_to_real) {
            std::valarray<T> image(_ncols * _nrows);
            input_raster.getBlock(image, 0, 0, _ncols, _nrows, band + 1);
            multilook(image, image_ml);
        } else {
            std::valarray<std::complex<T>> complex_image(_ncols * _nrows);
            input_raster.getBlock(complex_image, 0, 0, _ncols, _nrows,
                                  band + 1);
            multilook(complex_image, image_ml, exponent);
        }
        std::cout << "saving block" << std::endl;
        output_raster.setBlock(image_ml, 0, 0, _ncolsLooked, _nrowsLooked,
                               band + 1);
        std::cout << "...done" << std::endl;
    }
}

/**
 * * @param[in] input input array to be multi-looked
 * * @param[out] output output multilooked and downsampled array 
 * */
template<class T>
void isce3::signal::Looks<T>::multilook(std::valarray<T>& input,
                                       std::valarray<T>& output) {

    // Time-domain multi-looking of an array with following parameters
    // size of input array: _ncols * _nrows
    // number of looks on columns : _colsLooks
    // number of looks on rows : _rowsLooks
    // size of output array: _ncolsLooked * _nrowsLooked
    //
    // The mean of a box of size _colsLooks * _rowsLooks is computed
    //
    // This function is implemented to perform time domain
    // multi-looking in two steps:
    //  1) multi-looking in range direction (columns) and store the results in a
    //  temporary buffer 2) multi-looking the temporary array from previous step
    //  in azimuth direction (rows)

    // a temporary buffer to store the multi-looked data in range (columns)
    // direction
    std::valarray<T> tempOutput(_nrows * _ncolsLooked);

// multi-looking in range direction (columns)
#pragma omp parallel for
    for (size_t kk = 0; kk < _nrows * _ncolsLooked; ++kk) {
        size_t line = kk / _ncolsLooked;
        size_t col = kk % _ncolsLooked;
        T sum = 0.0;
        for (size_t j = col * _colsLooks; j < (col + 1) * _colsLooks; ++j) {
            sum += input[line * _ncols + j];
        }
        tempOutput[line * _ncolsLooked + col] = sum;
    }

// multi-looking in azimuth direction (rows)
#pragma omp parallel for
    for (size_t kk = 0; kk < _ncolsLooked * _nrowsLooked; ++kk) {
        size_t line = kk / _ncolsLooked;
        size_t col = kk % _ncolsLooked;
        T sum = 0.0;
        for (size_t i = line * _rowsLooks; i < (line + 1) * _rowsLooks; ++i) {
            sum += tempOutput[i * _ncolsLooked + col];
        }

        output[line * _ncolsLooked + col] = sum;
    }

    // To compute the mean
    output /= (_colsLooks * _rowsLooks);
}

/**
 * @param[in] input input array to be multi-looked
 * @param[out] output output multilooked and downsampled array
 * @param[in] noDataValue invalid data which will be excluded when multi-looking
 */
template<class T>
void isce3::signal::Looks<T>::multilook(std::valarray<T>& input,
                                       std::valarray<T>& output,
                                       T noDataValue) {

    // Multi-looking an array while taking into account the noDataValue.
    // Pixels whose value equals "noDataValue" is excluded in mult-looking.

    // a buffer for a boolean mask with the same size of the input array
    std::valarray<bool> mask(input.size());

    // create the mask. (mask[input==noDataValue] = false)
    mask = isce3::core::makeMask(input, noDataValue);

    // perform multi-looking using the mask array
    multilook(input, mask, output);
}

/**
 * @param[in] input input array to be multi-looked
 * @param[in] mask input boolean mask array to mask the input array before multi-looking 
 * @param[out] output output multilooked and downsampled array
 */
template<class T>
void isce3::signal::Looks<T>::multilook(std::valarray<T>& input,
                                       std::valarray<bool>& mask,
                                       std::valarray<T>& output) {

    // Multi-looking an array while taking into account a boolean mask.
    // Invalid pixels are excluded based on the mask.

    // a buffer for weights
    std::valarray<T> weights(input.size());

    // fill the weights array with zero
    weights = 0.0;

    // fill the weights for pixels with valid mask with one
    weights[mask] = 1.0;

    // multi-looking with the zero-one weight array
    multilook(input, weights, output);
}

/** 
 * @param[in] input input array to be multi-looked
 * @param[in] weights input weight array to weight the input array for multi-looking
 * @param[out] output output multilooked and downsampled array
 */
template<class T>
void isce3::signal::Looks<T>::multilook(std::valarray<T>& input,
                                       std::valarray<T>& weights,
                                       std::valarray<T>& output) {

    // A general implementation of multi-looking with weight array.

    // temporary buffers used for mult-looking columns for the
    // data and the weights
    std::valarray<T> tempOutput(_nrows * _ncolsLooked);
    std::valarray<T> tempSumWeights(_nrows * _ncolsLooked);

// weighted multi-looking the columns
#pragma omp parallel for
    for (size_t kk = 0; kk < _nrows * _ncolsLooked; ++kk) {
        size_t line = kk / _ncolsLooked;
        size_t col = kk % _ncolsLooked;
        T sum = 0.0;
        T sumWgt = 0;
        for (size_t j = col * _colsLooks; j < (col + 1) * _colsLooks; ++j) {
            sum += weights[line * _ncols + j] * input[line * _ncols + j];
            sumWgt += weights[line * _ncols + j];
        }
        tempOutput[line * _ncolsLooked + col] = sum;
        tempSumWeights[line * _ncolsLooked + col] = sumWgt;
    }

    // weighted multi-looking the rows
    #pragma omp parallel for
    for (size_t kk = 0; kk < _nrowsLooked*_ncolsLooked; ++kk){
        size_t line = kk/_ncolsLooked;
        size_t col = kk%_ncolsLooked;

        T sum = 0.0;
        T sumWgt = 0;
        for (size_t i=line*_rowsLooks; i<(line+1)*_rowsLooks; ++i){

            // Note that the elements of tempOutput are already weighted in the previous loop. 
            // So no need to weight them again.
            sum += tempOutput[i*_ncolsLooked + col];
            sumWgt += tempSumWeights[i*_ncolsLooked + col];
        }

        // To avoid dividing by zero
        if (sumWgt>0)
            output[line*_ncolsLooked + col] = sum/sumWgt;
    }
}

/**
 * @param[in] input input array of complex data to be multi-looked
 * @param[out] output output multilooked and downsampled array of complex data
 */
template<class T>
void isce3::signal::Looks<T>::multilook(std::valarray<std::complex<T>>& input,
                                       std::valarray<std::complex<T>>& output) {

    // The implementation details are same as real data. See the notes above.

    std::valarray<std::complex<T>> tempOutput(_nrows * _ncolsLooked);

#pragma omp parallel for
    for (size_t kk = 0; kk < _nrows * _ncolsLooked; ++kk) {
        size_t line = kk / _ncolsLooked;
        size_t col = kk % _ncolsLooked;
        std::complex<T> sum = std::complex<T>(0.0, 0.0);
        for (size_t j = col * _colsLooks; j < (col + 1) * _colsLooks; ++j) {
            sum += input[line * _ncols + j];
        }
        tempOutput[line * _ncolsLooked + col] = sum;
    }

#pragma omp parallel for
    for (size_t kk = 0; kk < _nrowsLooked * _ncolsLooked; ++kk){
        size_t line = kk/_ncolsLooked;
        size_t col = kk%_ncolsLooked;
        std::complex<T> sum = std::complex<T> (0.0 , 0.0);
        for (size_t i=line*_rowsLooks; i<(line+1)*_rowsLooks; ++i){
            sum += tempOutput[i*_ncolsLooked + col];
        }
        output[line * _ncolsLooked + col] = sum;
    }

    output /= (_colsLooks * _rowsLooks);
}

/**
 * @param[in] input input array of complex data to be multi-looked
 * @param[out] output output multilooked and downsampled array of complex data
 * @param[out] noDataValue invalid complex data which will be excluded when multi-looking
 */
template <class T>
void
isce3::signal::Looks<T>::
multilook(std::valarray<std::complex<T>> &input,
            std::valarray<std::complex<T>> &output,
            std::complex<T> noDataValue)
{

    // buffer for a boolean mask
    std::valarray<bool> mask(input.size());

    // create the mask. (mask[input==noDataValue] = false)
    mask = isce3::core::makeMask(input, noDataValue);

    // multilooking 
    multilook(input, mask, output);

}

/**
 * @param[in] input input array of complex data to be multi-looked
 * @param[in] mask input boolean mask array to mask the input array before multi-looking 
 * @param[out] noDataValue invalid complex data which will be excluded when multi-looking
 */
template <class T>
void
isce3::signal::Looks<T>::
multilook(std::valarray<std::complex<T>> &input,
        std::valarray<bool> &mask,
        std::valarray<std::complex<T>> &output)
{

    std::valarray<T> weights(input.size());
    weights = 0.0;
    weights[mask] = 1.0;
    multilook(input, weights, output);

}

/**
 * @param[in] input input array of complex data to be multi-looked
 * @param[in] weights input weight array to weight the input array for multi-looking
 * @param[out] noDataValue invalid complex data which will be excluded when multi-looking
 */
template <class T>
void
isce3::signal::Looks<T>::
multilook(std::valarray<std::complex<T>> &input,
            std::valarray<T> &weights,
            std::valarray<std::complex<T>> &output)
{

    std::valarray<std::complex<T>> tempOutput(_nrows*_ncolsLooked);
    std::valarray<T> tempSumWeights(_nrows * _ncolsLooked);

#pragma omp parallel for
    for (size_t kk = 0; kk < _nrows*_ncolsLooked; ++kk){
        size_t line = kk/_ncolsLooked;
        size_t col = kk%_ncolsLooked;

        std::complex<T> sum = std::complex<T> (0.0, 0.0);
        T sumWeights = 0;

        for (size_t j=col*_colsLooks; j<(col+1)*_colsLooks; ++j){
            sum += weights[line*_ncols+j] * input[line*_ncols+j];
            sumWeights += weights[line * _ncols + j];
        }

        tempOutput[line*_ncolsLooked + col] = sum;
        tempSumWeights[line * _ncolsLooked + col] = sumWeights;
    }

    #pragma omp parallel for
    for (size_t kk = 0; kk < _nrowsLooked * _ncolsLooked; ++kk){
        size_t line = kk/_ncolsLooked;
        size_t col = kk%_ncolsLooked;
        std::complex<T> sum = std::complex<T> (0.0 , 0.0);
        T sumWeights = 0;
        for (size_t i=line*_rowsLooks; i<(line+1)*_rowsLooks; ++i){
            sum += tempOutput[i*_ncolsLooked + col];
            sumWeights += tempSumWeights[i * _ncolsLooked + col];
        }

        output[line * _ncolsLooked + col] = sum / sumWeights;
    }
    
}

/**
 * @param[in] input input array of complex data
 * @param[out] output output array of real data
 * @param[in] exponent the power to which the absolute of complex data are
 * raisen to before multi-looking
 */
template<class T>
void isce3::signal::Looks<T>::multilook(std::valarray<std::complex<T>>& input,
                                       std::valarray<T>& output, int exponent) {

    // If exponent == 0, apply default complex-to-float multilooking (squared)
    if (exponent == 0)
        exponent = 2;

    std::valarray<T> tempOutput(_nrows * _ncolsLooked);

#pragma omp parallel for
    for (size_t kk = 0; kk < _nrows * _ncolsLooked; ++kk) {
        size_t line = kk / _ncolsLooked;
        size_t col = kk % _ncolsLooked;
        T sum = 0.0;
        for (size_t j = col * _colsLooks; j < (col + 1) * _colsLooks; ++j) {
            sum += std::pow(std::abs(input[line * _ncols + j]), exponent);
        }
        tempOutput[line * _ncolsLooked + col] = sum;
    }

#pragma omp parallel for
    for (size_t kk = 0; kk < _nrowsLooked * _ncolsLooked; ++kk){
        size_t line = kk/_ncolsLooked;
        size_t col = kk%_ncolsLooked;
        T sum = 0.0;
        for (size_t i=line*_rowsLooks; i<(line+1)*_rowsLooks; ++i){
            sum += tempOutput[i*_ncolsLooked + col];
        }

        output[line*_ncolsLooked+col] = sum;
    }
    output /= (_colsLooks * _rowsLooks);
}

template class isce3::signal::Looks<float>;
template class isce3::signal::Looks<double>;

