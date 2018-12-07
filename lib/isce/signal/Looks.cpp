// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2018-
//

#include "Looks.h"



template <class T>
void
isce::signal::Looks<T>::
multilook(std::valarray<T> &input,
        std::valarray<T> &output)
{

    std::valarray<T> tempOutput(_nrows*_ncolsLooked);

    for (size_t line=0; line<_nrows; ++line){
        for (size_t col=0; col<_ncolsLooked; ++col){
            T sum = 0.0;
            for (size_t j=col*_colsLooks; j<(col+1)*_colsLooks; ++j){
                sum += input[line*_ncols+j];
            }
            tempOutput[line*_ncolsLooked + col] = sum;
        }

    }

    for (size_t col=0; col<_ncolsLooked; ++col){
        for (size_t line=0; line<_nrowsLooked; ++line){
            T sum = 0.0;
            for (size_t i=line*_rowsLooks; i<(line+1)*_rowsLooks; ++i){
                sum += tempOutput[i*_ncolsLooked + col];
            }
            
            output[line*_ncolsLooked+col] = sum;
        }
    }
    output /= (_colsLooks*_rowsLooks);
}

template <class T>
void
isce::signal::Looks<T>::
multilook(std::valarray<T> &input,
	std::valarray<bool> &mask, 
        std::valarray<T> &output)
{

    std::valarray<T> tempOutput(_nrows*_ncolsLooked);
    std::valarray<int> tempCounterMask(_nrows*_ncolsLooked);
    for (size_t line=0; line<_nrows; ++line){
        for (size_t col=0; col<_ncolsLooked; ++col){
            T sum = 0.0;
            int counter = 0;
            for (size_t j=col*_colsLooks; j<(col+1)*_colsLooks; ++j){
		if (mask[line*_ncols+j]){
                    sum += input[line*_ncols+j];
                    counter += 1;
                }
            }
            tempOutput[line*_ncolsLooked + col] = sum;
            tempCounterMask[line*_ncolsLooked + col] = counter;
        }

    }

    T minCounter = 1.0;
    for (size_t col=0; col<_ncolsLooked; ++col){
        for (size_t line=0; line<_nrowsLooked; ++line){
            T sum = 0.0;
            T counter = 0;
            for (size_t i=line*_rowsLooks; i<(line+1)*_rowsLooks; ++i){
                sum += tempOutput[i*_ncolsLooked + col];
                counter += tempCounterMask[i*_ncolsLooked + col];
            }
            
            output[line*_ncolsLooked + col] = sum/std::max(minCounter, counter);
        }
    }

}

template <class T>
void
isce::signal::Looks<T>::
multilook(std::valarray<T> &input,
        std::valarray<T> &weights,
        std::valarray<T> &output)
{

    std::valarray<T> tempOutput(_nrows*_ncolsLooked);
    std::valarray<T> tempSumWeights(_nrows*_ncolsLooked);
    for (size_t line=0; line<_nrows; ++line){
        for (size_t col=0; col<_ncolsLooked; ++col){
            T sum = 0.0;
            T sumWgt = 0;
            for (size_t j=col*_colsLooks; j<(col+1)*_colsLooks; ++j){
                sum += weights[line*_ncols+j]*input[line*_ncols+j];
                sumWgt += weights[line*_ncols+j];
            }
            tempOutput[line*_ncolsLooked + col] = sum;
            tempSumWeights[line*_ncolsLooked + col] = sumWgt;
        }

    }

    for (size_t col=0; col<_ncolsLooked; ++col){
        for (size_t line=0; line<_nrowsLooked; ++line){
            T sum = 0.0;
            T sumWgt = 0;
            for (size_t i=line*_rowsLooks; i<(line+1)*_rowsLooks; ++i){

                sum += tempSumWeights[i*_ncolsLooked + col]*
                        tempOutput[i*_ncolsLooked + col];

                sumWgt += tempSumWeights[i*_ncolsLooked + col];
            }

            // To avoid dividing by zero
            // Can we avoid this if statement?
            if (sumWgt>0)
                output[line*_ncolsLooked + col] = sum/sumWgt; //std::max(sumWgt, counter);
        }
    }

}


template <class T>
void
isce::signal::Looks<T>::
multilook(std::valarray<std::complex<T>> &input,
        std::valarray<std::complex<T>> &output)
{

    std::valarray<std::complex<T>> tempOutput(_nrows*_ncolsLooked);

    for (size_t line=0; line<_nrows; ++line){
        for (size_t col=0; col<_ncolsLooked; ++col){
            std::complex<T> sum = std::complex<T> (0.0, 0.0);
            for (size_t j=col*_colsLooks; j<(col+1)*_colsLooks; ++j){
                sum += input[line*_ncols+j];
            }
            tempOutput[line*_ncolsLooked + col] = sum;
        }

    }

    for (size_t col=0; col<_ncolsLooked; ++col){
        for (size_t line=0; line<_nrowsLooked; ++line){
            std::complex<T> sum = std::complex<T> (0.0 , 0.0);
            for (size_t i=line*_rowsLooks; i<(line+1)*_rowsLooks; ++i){
                sum += tempOutput[i*_ncolsLooked + col];
            }

            output[line*_ncolsLooked+col] = sum;
        }
    }
}

template <class T>
void
isce::signal::Looks<T>::
multilook(std::valarray<std::complex<T>> &input,
        std::valarray<T> &output, int p)
{

    std::valarray<T> tempOutput(_nrows*_ncolsLooked);

    for (size_t line=0; line<_nrows; ++line){
        for (size_t col=0; col<_ncolsLooked; ++col){
            T sum = 0.0;
            for (size_t j=col*_colsLooks; j<(col+1)*_colsLooks; ++j){
                //sum += std::pow(std::abs(input[line*_ncols+j]), p);
                sum += std::abs(input[line*_ncols+j])*std::abs(input[line*_ncols+j]);
            }
            tempOutput[line*_ncolsLooked + col] = sum;
        }

    }

    for (size_t col=0; col<_ncolsLooked; ++col){
        for (size_t line=0; line<_nrowsLooked; ++line){
            T sum = 0.0;
            for (size_t i=line*_rowsLooks; i<(line+1)*_rowsLooks; ++i){
                //sum += std::pow(std::abs(tempOutput[i*_ncolsLooked + col]), p);
                sum += tempOutput[i*_ncolsLooked + col];
            }

            output[line*_ncolsLooked+col] = sum;
        }
    }
}

template class isce::signal::Looks<float>;
template class isce::signal::Looks<double>;

