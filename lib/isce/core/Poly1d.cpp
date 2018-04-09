//
// Author: Joshua Cohen
// Copyright 2017
//

//#include <portinfo>
#include <iostream>
#include <stdexcept>
#include <pyre/journal.h>
#include "Poly1d.h"

double isce::core::Poly1d::
eval(double xin) const {
    /*
     * Evaluate the polynomial at a given position.
     */

    // Throw an exception if class member norm has value 0
    if (norm == 0.) {
            // make a channel
            pyre::journal::firewall_t channel("isce.core.domain");
            // complain
            channel
                << pyre::journal::at(__HERE__)
                << "divide by zero domain_error: norm == 0"
                << pyre::journal::endl;
            // and bail
            return 1;
    }

    double val = 0.;
    double scalex = 1.;
    double xmod = (xin - mean) / norm;
    for (int i=0; i<=order; i++,scalex*=xmod) val += scalex * coeffs[i];
    return val;
}

void isce::core::Poly1d::
printPoly() const {
    std::cout << "Polynomial Order: " << order << std::endl;
    for (int i=0; i<=order; i++) std::cout << getCoeff(i) << " ";
    std::cout << std::endl;
}

isce::core::Poly1d isce::core::Poly1d::
derivative() const {
    /*
     * Helper function to adjust the mean.
     * Use case - when image is being cropped and starting range is changed.
     */
    // If the input polynomial is a constant, return 0
    if (order == 0) {
        Poly1d newP(0, 0., 1.);
        newP.setCoeff(0, 0.);
        return newP;
    } else {
        // Initialize polynomial of same size
        Poly1d newP(order-1, mean, norm);
        for (int ii=0; ii<order; ii++) {
            double coeff = getCoeff(ii+1);
            newP.setCoeff(ii, ((ii + 1.) * coeff) / norm);
        }
        return newP;
    }
}

// end of file
