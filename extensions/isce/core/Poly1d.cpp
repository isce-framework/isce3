//
// Author: Joshua Cohen
// Copyright 2017
//

#include <iostream>
#include "isce/core/Poly1d.h"
using isce::core::Poly1d;
using std::cout;
using std::endl;


double Poly1d::eval(double xin) const {
    // Evaluate the polynomial at a given position

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

