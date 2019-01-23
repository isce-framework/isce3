//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2019-

#include "Interpolator.h"

template <typename U>
isce::core::BicubicInterpolator<U>::
BicubicInterpolator() : isce::core::Interpolator<U>(isce::core::BICUBIC_METHOD) {
    // Set the weights of the 2D bicubic kernel
    // This is an implementation of the bicubic interpolation based on:
    // http://www.ahinson.com/algorithms_general/Sections/InterpolationRegression/InterpolationBicubic.pdf
    // Given 16 data pints (4x4 neighborhood), the interpolated value 
    // at a new point can becomputed as:
    // P = [Py].[Y].[Z].[X].[Px]
    // where Px, Py are 4x1 and 1x4 vectors
    // Z is the data points surrounding the point of interest,
    // X,Y are the following 4x4 matrioces:
    _X.resize(4,4);
    _Y.resize(4,4);

    _X(0,0) = -1.0/6.0;
    _X(0,1) = 1.0/2.0;
    _X(0,2) = -1.0/3.0;
    _X(0,3) = 0.0;

    _X(1,0) = 1.0/2.0;
    _X(1,1) = -1.0;
    _X(1,2) = -1.0/2.0;
    _X(1,3) = 1.0;

    _X(2,0) = -1.0/2.0;
    _X(2,1) = 1.0/2.0;
    _X(2,2) = 1.0;
    _X(2,3) = 0.0;

    _X(3,0) = 1.0/6.0;
    _X(3,1) = 0.0;
    _X(3,2) = -1.0/6.0;
    _X(3,3) = 0.0;
   
    _Y(0,0) = -1.0/6.0;
    _Y(0,1) = 1.0/2.0;
    _Y(0,2) = -1.0/2.0;
    _Y(0,3) = 1.0/6.0;

    _Y(1,0) = 1.0/2.0;
    _Y(1,1) = -1.0;
    _Y(1,2) = 1.0/2.0;
    _Y(1,3) = 0.0;

    _Y(2,0) = -1.0/3.0;
    _Y(2,1) = -1.0/2.0;
    _Y(2,2) = 1.0;
    _Y(2,3) = -1.0/6.0;

    _Y(3,0) = 0.0;
    _Y(3,1) = 1.0;
    _Y(3,2) = 0.0;
    _Y(3,3) = 0.0;

    
}

/** @param[in] x X-coordinate to interpolate
  * @param[in] y Y-coordinate to interpolate
  * @param[in] z 2D matrix to interpolate. */
template <class U>
U
isce::core::BicubicInterpolator<U>::
interpolate(double x, double y, const isce::core::Matrix<U> & z) {

    // the closest pixel to the point of interest
    const int j = std::floor(x) ;
    const int i = std::floor(y) ;

    // fraction relative to j,i
    double px = x - j;
    double py = y - i;

    // the data matrix surrounding the interpolation point
    isce::core::Matrix<U>  Z(4,4);
    
    Z(0,0) = z(i-1, j-1);      //row0 col0 >> x0,y0
    Z(0,1) = z(i-1, j);         //row0 col1 >> x1,y0
    Z(0,2) = z(i-1, j+1);       //row0 col2 >> x2,y0

    Z(1,0) = z(i, j-1);         //row1 col0 >> x0,y1
    Z(1,1) = z(i, j);          //row1 col1 >> x1,y1
    Z(1,2) = z(i, j+1);        //row1 col2 >> x2,y1

    Z(2,0) = z(i+1,j-1);       //row2 col0 >> x0,y2
    Z(2,1) = z(i+1,j);    //row2 col1 >> x1,y2
    Z(2,2) = z(i+1,j+1);  //row2 col2 >> x2,y2

    Z(0,3) = 0;               //row0 col3 >> x3,y0
    Z(1,3) = 0;               //row1 col3 >> x3,y1
    Z(2,3) = 0;               //row2 col3 >> x3,y2
    Z(3,0) = 0;               //row3 col0 >> x0,y3
    Z(3,1) = 0;               //row3 col1 >> x1,y3
    Z(3,2) = 0;               //row3 col2 >> x2,y3
    Z(3,3) = 0;               //row3 col3 >> x3,y3


    if (i > 0 && i < (z.length() - 2) && j > 0 && j < (z.width()-2)) {
        
         Z(0,3) = z(i-1, j+2);      //row0 col3 >> x3,y0
         Z(1,3) = z(i,j+2);         //row1 col3 >> x3,y1
         Z(2,3) = z(i+1,j+2);       //row2 col3 >> x3,y2
         Z(3,0) = z(i+2,j-1);      //row3 col0 >> x0,y3
         Z(3,1) = z(i+2,j);         //row3 col1 >> x1,y3
         Z(3,2) = z(i+2,j+1);       //row3 col2 >> x2,y3
         Z(3,3) = z(i+2,j+2);       //row3 col3 >> x3,y3

    }
    else if (i < 0) {

         Z(0,3) = Z(0,2);               //row0 col3 >> x3,y0
         Z(1,3) = Z(1,2);               //row1 col3 >> x3,y1
         Z(2,3) = Z(2,2);               //row2 col3 >> x3,y2
         Z(3,0) = z(i+2,j-1);       //row3 col0 >> x0,y3
         Z(3,1) = z(i+2,j);         //row3 col1 >> x1,y3
         Z(3,2) = z(i+2,j+1);       //row3 col2 >> x2,y3
         Z(3,3) = Z(3,2);              //row3 col3 >> x3,y3

    }
    else if (j < 0 ) {

         Z(0,3) = z(i-1, j+2);      //row0 col3 >> x3,y0
         Z(1,3) = z(i,j+2);         //row1 col3 >> x3,y1
         Z(2,3) = z(i+1,j+2);       //row2 col3 >> x3,y2
         Z(3,0) = Z(2,0);               //row3 col0 >> x0,y3
         Z(3,1) = Z(2,1);               //row3 col1 >> x1,y3
         Z(3,2) = Z(2,2);               //row3 col2 >> x2,y3
         Z(3,3) = Z(2,3);               //row3 col3 >> x3,y3
    } 
    else if (i == (z.length() - 2) && j == (z.width() - 2) ) {

         Z(0,3) = Z(0,2);               //row0 col3 >> x3,y0
         Z(1,3) = Z(1,2);               //row1 col3 >> x3,y1
         Z(2,3) = Z(2,2);               //row2 col3 >> x3,y2
         Z(3,0) = Z(2,0);               //row3 col0 >> x0,y3
         Z(3,1) = Z(2,1);               //row3 col1 >> x1,y3
         Z(3,2) = Z(2,2);               //row3 col2 >> x2,y3
         Z(3,3) = Z(2,3);               //row3 col3 >> x3,y3
    
    }
    
    // required vectors for qubic interpolation over columns and rows
    std::valarray<double>  Px(4);
    std::valarray<double>  Py(4); 
    
    Px[0] = std::pow(px, 3.0);
    Px[1] = std::pow(px, 2.0);
    Px[2] = px;
    Px[3] = 1.0;
    
    Py[0] = std::pow(py, 3.0);
    Py[1] = std::pow(py, 2.0);
    Py[2] = py;
    Py[3] = 1.0;
    
    // get elements of [X].[Px]
    U g0 = static_cast<U> (Px[0]*_X(0,0) + Px[1]*_X(0,1) + Px[2]*_X(0,2) + Px[3]*_X(0,3));
    U g1 = static_cast<U> (Px[0]*_X(1,0) + Px[1]*_X(1,1) + Px[2]*_X(1,2) + Px[3]*_X(1,3));
    U g2 = static_cast<U> (Px[0]*_X(2,0) + Px[1]*_X(2,1) + Px[2]*_X(2,2) + Px[3]*_X(2,3));
    U g3 = static_cast<U> (Px[0]*_X(3,0) + Px[1]*_X(3,1) + Px[2]*_X(3,2) + Px[3]*_X(3,3));
    
    // Given all required matrices and cetors, the value at position x,y
    // can be obtained as:
    // P = [Py].[Y].[Z].[X].[Px]
    U P = static_cast<U> (Py[0]*_Y(0,0) + Py[1]*_Y(1,0) + Py[2]*_Y(2,0) + Py[3]*_Y(3,0)) *
                (Z(0,0)*g0 + Z(0,1)*g1 + Z(0,2)*g2 + Z(0,3)*g3) +    
            static_cast<U> (Py[0]*_Y(0,1) + Py[1]*_Y(1,1) + Py[2]*_Y(2,1) + Py[3]*_Y(3,1)) *
                (Z(1,0)*g0 + Z(1,1)*g1 + Z(1,2)*g2 + Z(1,3)*g3) +
            static_cast<U> (Py[0]*_Y(0,2) + Py[1]*_Y(1,2) + Py[2]*_Y(2,2) + Py[3]*_Y(3,2)) *
                (Z(2,0)*g0 + Z(2,1)*g1 + Z(2,2)*g2 + Z(2,3)*g3) +
            static_cast<U> (Py[0]*_Y(0,3) + Py[1]*_Y(1,3) + Py[2]*_Y(2,3) + Py[3]*_Y(3,3)) *
                (Z(3,0)*g0 + Z(3,1)*g1 + Z(3,2)*g2 + Z(3,3)*g3);

    return P;
       
}

// Forward declaration of classes
template class isce::core::BicubicInterpolator<double>;
template class isce::core::BicubicInterpolator<float>;
template class isce::core::BicubicInterpolator<std::complex<double>>;
template class isce::core::BicubicInterpolator<std::complex<float>>;

// end of file
