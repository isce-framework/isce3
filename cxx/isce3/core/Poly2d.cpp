#include <iostream>
#include "Constants.h"
#include "Poly2d.h"

/**
 * @param[in] azi azimuth or y value
 * @param[in] rng range or x value*/
double isce3::core::Poly2d::
eval(double y, double x) const {

    double xval = (x - xMean) / xNorm;
    double yval = (y - yMean) / yNorm;

    double scalex;
    double scaley = 1.;
    double val = 0.;
    for (int i=0; i<=yOrder; i++,scaley*=yval) {
        scalex = 1.;
        for (int j=0; j<=xOrder; j++,scalex*=xval) {
            val += scalex * scaley * coeffs[IDX1D(i,j,xOrder+1)];
        }
    }
    return val;
}

void isce3::core::Poly2d::
printPoly() const {
    std::cout << "Polynomial Order: " << yOrder << " - by - " << xOrder << std::endl;
    for (int i=0; i<=yOrder; i++) {
        for (int j=0; j<=xOrder; j++) {
            std::cout << getCoeff(i,j) << " ";
        }
        std::cout << std::endl;
    }
}
