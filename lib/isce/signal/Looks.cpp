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

    std::valarray<T> tempOutput(_nrows*_ncolsLooked);
    
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
    
    std::valarray<bool> mask(input.size());
    mask = true;
    mask[input==noDataValue] = false;
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

    std::valarray<T> weights(input.size());
    weights = 0.0;
    weights[mask] = 1.0;
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

    std::valarray<T> tempOutput(_nrows*_ncolsLooked);
    std::valarray<T> tempSumWeights(_nrows*_ncolsLooked);

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

    #pragma omp parallel for
    for (size_t kk = 0; kk < _nrowsLooked*_ncolsLooked; ++kk){
        size_t line = kk/_ncolsLooked;
        size_t col = kk%_ncolsLooked;

        T sum = 0.0;
        T sumWgt = 0;
        for (size_t i=line*_rowsLooks; i<(line+1)*_rowsLooks; ++i){

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

    std::valarray<bool> mask(input.size());
    mask = true;
    mask[input == noDataValue] = false;
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

