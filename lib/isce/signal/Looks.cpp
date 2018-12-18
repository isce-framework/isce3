// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2018-
//

#include "Looks.h"

/**
 * * @param[in] input input array to be multi-looked
 * * @param[out] output output multilooked and downsampled array 
 * */
template <class T>
void
isce::signal::Looks<T>::
multilook(std::valarray<T> &input,
        std::valarray<T> &output)
{

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
    //  1) multi-looking in range direction (columns) and store the results in a temporary buffer
    //  2) multi-looking the temporary array from previous step in azimuth direction (rows) 

    // a temporary buffer to store the multi-looked data in range (columns) direction
    std::valarray<T> tempOutput(_nrows*_ncolsLooked);
   
    // multi-lokking in range direction (columns)
    #pragma omp parallel for
    for (size_t kk = 0; kk < _nrows*_ncolsLooked; ++kk){
        size_t line = kk/_ncolsLooked;
        size_t col = kk%_ncolsLooked;
        T sum = 0.0;
        for (size_t j=col*_colsLooks; j<(col+1)*_colsLooks; ++j){
            sum += input[line*_ncols+j];
        }
        tempOutput[line*_ncolsLooked + col] = sum;
    }

    // multi-lokking in azimuth direction (rows)
    #pragma omp parallel for
    for (size_t kk = 0; kk < _ncolsLooked*_nrowsLooked; ++kk){
        size_t line = kk/_ncolsLooked ;
        size_t col = kk%_ncolsLooked;
        T sum = 0.0;
        for (size_t i=line*_rowsLooks; i<(line+1)*_rowsLooks; ++i){
            sum += tempOutput[i*_ncolsLooked + col];
        }

        output[line*_ncolsLooked+col] = sum;
    }
 
    // To compute the mean
    output /= (_colsLooks*_rowsLooks);
}

/**
 * @param[in] input input array to be multi-looked
 * @param[out] output output multilooked and downsampled array
 * @param[in] noDataValue invalid data which will be excluded when multi-looking
 */
template <class T>
void
isce::signal::Looks<T>::
multilook(std::valarray<T> &input,
            std::valarray<T> &output,
            T noDataValue)
{
   
    // Multi-looking an array while taking into account the noDataValue.
    // Pixels whose value equals "noDataValue" is excluded in mult-looking.

    // a buffer for a boolean mask with the same size of the input array 
    std::valarray<bool> mask(input.size());

    // create the mask. (mask[input==noDataValue] = false) 
    mask = isce::core::makeMask(input, noDataValue);

    // perform multi-looking using the mask array
    multilook(input, mask, output);

}

/**
 * @param[in] input input array to be multi-looked
 * @param[in] mask input boolean mask array to mask the input array before multi-looking 
 * @param[out] output output multilooked and downsampled array
 */
template <class T>
void
isce::signal::Looks<T>::
multilook(std::valarray<T> &input,
        std::valarray<bool> &mask,
        std::valarray<T> &output)
{

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
template <class T>
void
isce::signal::Looks<T>::
multilook(std::valarray<T> &input,
        std::valarray<T> &weights,
        std::valarray<T> &output)
{

    // A general implementation of multi-looking with weight array.

    // temporary buffers used for mult-looking columns for the 
    // data and the weights
    std::valarray<T> tempOutput(_nrows*_ncolsLooked);
    std::valarray<T> tempSumWeights(_nrows*_ncolsLooked);

    // weighted multi-looking the columns 
    #pragma omp parallel for
    for (size_t kk = 0; kk < _nrows*_ncolsLooked; ++kk){
        size_t line = kk/_ncolsLooked;
        size_t col = kk%_ncolsLooked;
        T sum = 0.0;
        T sumWgt = 0;
        for (size_t j=col*_colsLooks; j<(col+1)*_colsLooks; ++j){
            sum += weights[line*_ncols+j]*input[line*_ncols+j];
            sumWgt += weights[line*_ncols+j];
        }
        tempOutput[line*_ncolsLooked + col] = sum;
        tempSumWeights[line*_ncolsLooked + col] = sumWgt;
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
template <class T>
void
isce::signal::Looks<T>::
multilook(std::valarray<std::complex<T>> &input,
        std::valarray<std::complex<T>> &output)
{

    // The implementation details are same as real data. See the notes above.

    std::valarray<std::complex<T>> tempOutput(_nrows*_ncolsLooked);
    
    #pragma omp parallel for
    for (size_t kk = 0; kk < _nrows*_ncolsLooked; ++kk){
        size_t line = kk/_ncolsLooked;
        size_t col = kk%_ncolsLooked;
        std::complex<T> sum = std::complex<T> (0.0, 0.0);
        for (size_t j=col*_colsLooks; j<(col+1)*_colsLooks; ++j){
            sum += input[line*_ncols+j];
        }
        tempOutput[line*_ncolsLooked + col] = sum;
    }
    
    #pragma omp parallel for
    for (size_t kk = 0; kk < _nrowsLooked * _ncolsLooked; ++kk){
        size_t line = kk/_ncolsLooked;
        size_t col = kk%_ncolsLooked;
        std::complex<T> sum = std::complex<T> (0.0 , 0.0);
        for (size_t i=line*_rowsLooks; i<(line+1)*_rowsLooks; ++i){
            sum += tempOutput[i*_ncolsLooked + col];
        }

        output[line*_ncolsLooked+col] = sum;
    }
   
}

/**
 * @param[in] input input array of complex data to be multi-looked
 * @param[out] output output multilooked and downsampled array of complex data
 * @param[out] noDataValue invalid complex data which will be excluded when multi-looking
 */
template <class T>
void
isce::signal::Looks<T>::
multilook(std::valarray<std::complex<T>> &input,
            std::valarray<std::complex<T>> &output,
            std::complex<T> noDataValue)
{

    // buffer for a boolean mask
    std::valarray<bool> mask(input.size());

    // create the mask. (mask[input==noDataValue] = false)
    mask = isce::core::makeMask(input, noDataValue);

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
isce::signal::Looks<T>::
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
isce::signal::Looks<T>::
multilook(std::valarray<std::complex<T>> &input,
            std::valarray<T> &weights,
            std::valarray<std::complex<T>> &output)
{

    std::valarray<std::complex<T>> tempOutput(_nrows*_ncolsLooked);
    
    #pragma omp parallel for
    for (size_t kk = 0; kk < _nrows*_ncolsLooked; ++kk){
        size_t line = kk/_ncolsLooked;
        size_t col = kk%_ncolsLooked;

        std::complex<T> sum = std::complex<T> (0.0, 0.0);
            
        for (size_t j=col*_colsLooks; j<(col+1)*_colsLooks; ++j){
            sum += weights[line*_ncols+j] * input[line*_ncols+j];
        }

        tempOutput[line*_ncolsLooked + col] = sum;
    }

    #pragma omp parallel for
    for (size_t kk = 0; kk < _nrowsLooked * _ncolsLooked; ++kk){
        size_t line = kk/_ncolsLooked;
        size_t col = kk%_ncolsLooked;
        std::complex<T> sum = std::complex<T> (0.0 , 0.0);
        for (size_t i=line*_rowsLooks; i<(line+1)*_rowsLooks; ++i){
            sum += tempOutput[i*_ncolsLooked + col];
        }

        output[line*_ncolsLooked+col] = sum;
    }
    
}

/**
 * @param[in] input input array of complex data
 * @param[out] output output array of real data
 * @param[in] p exponent, the power to which the absolute of complex data are raisen to before multi-looking  
*/
template <class T>
void
isce::signal::Looks<T>::
multilook(std::valarray<std::complex<T>> &input,
        std::valarray<T> &output, int p)
{

    std::valarray<T> tempOutput(_nrows*_ncolsLooked);

    #pragma omp parallel for
    for (size_t kk = 0; kk < _nrows*_ncolsLooked; ++kk){
        size_t line = kk/_ncolsLooked;
        size_t col = kk%_ncolsLooked;
        T sum = 0.0;
        for (size_t j=col*_colsLooks; j<(col+1)*_colsLooks; ++j){
            sum += std::pow(std::abs(input[line*_ncols+j]), p);
        }
        tempOutput[line*_ncolsLooked + col] = sum;
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
    
}

template class isce::signal::Looks<int>;
template class isce::signal::Looks<float>;
template class isce::signal::Looks<double>;

