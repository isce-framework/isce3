//
// Author: Joshua Cohen
// Copyright 2017
//

#include <iostream>
#include "Poly1d.h"
using isce::core::Poly1d;
using std::cout;
using std::endl;


double Poly1d::eval(double xin) const {
    // Evaluate the polynomial at a given position

    // throw an exception if class member norm has value 0
    if ( norm == 0. ) {
        throw std::overflow_error("Poly1d::eval norm==0.: Divide by zero exception");
    }

    double val = 0.;
    double scalex = 1.;
    double xmod = (xin - mean) / norm;
    for (int i=0; i<=order; i++,scalex*=xmod) val += scalex * coeffs[i];
    return val;
}

void Poly1d::printPoly() const {
    cout << "Polynomial Order: " << order << endl;
    for (int i=0; i<=order; i++) cout << getCoeff(i) << " ";
    cout << endl;
}


//Helper function to adjust the mean
//Use case - when image is being cropped and starting range is changed
Poly1d Poly1d::derivative() const
{
    //If the input polynomial is a constant, return 0
    if (order == 0)
    {
        Poly1d newP(0, 0., 1.);
        newP.setCoeff(0, 0.0);

        return newP;
    }
    else
    {
        //Initialize polynomial of same size
        Poly1d newP(order-1, mean, norm);
        for(int ii=0; ii < order; ii++)
        {
            double coeff = getCoeff(ii+1);
            newP.setCoeff(ii, (ii+1.0) * coeff / norm);
        }

        return newP;
    }
}
