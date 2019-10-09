// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#include <iostream>
#include <gsl/gsl_rng.h>

int main(int, char*[]) {
    // the sample size
    const int N = 1.0e7;
    // initialize the counters */
    int interior = 0, total = 0;
    // create the random number generator
    gsl_rng * generator = gsl_rng_alloc(gsl_rng_ranlxs2);
    // integrate by sampling some number of times
    for (int i=0; i<N; i++) {
        // create a random point
        double x = gsl_rng_uniform(generator);
        double y = gsl_rng_uniform(generator);
        // check whether it is inside the unit quarter circle
        if ((x*x + y*y) <= 1.0) { // no need to waste time computing the square root
            // update the interior point counter
            interior++;
        }
        // update the total number of points
        total++;
    }
    // print the result
    std::cout << "pi: " << 4. * interior / N << std::endl;
    return 0;
}

// end of file
