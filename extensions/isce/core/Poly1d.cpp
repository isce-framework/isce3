//
// Author: Joshua Cohen
// Copyright 2017
//

#include <iostream>
#include "Poly1d.h"
using std::cout;
using std::endl;
using isceLib::Poly1d;


double Poly1d::eval(double xin) {
    // Evaluate the polynomial at a given position

    double val = 0.;
    double scalex = 1.;
    auto xmod = (xin - mean) / norm;
    for (int i=0; i<=order; i++,scalex*=xmod) val += scalex * coeffs[i];
    return val;
}

void Poly1d::printPoly() {
    cout << "Polynomial Order: " << order << endl;
    for (int i=0; i<=order; i++) cout << getCoeff(i) << " ";
    cout << endl;
}

