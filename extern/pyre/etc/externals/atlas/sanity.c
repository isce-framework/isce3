// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// get the headers
#include <cblas.h>


// smallest possible driver
int main() {
    // allocate the scala
    double alpha = 1;
    // allocate a vector
    double x[] = {0.0, 0.0, 0.0};
    double y[] = {0.0, 0.0, 0.0};

    // do it
    cblas_daxpy(3, alpha, x, 1, y, 1);

    // all done
    return 0;
}

// end of file
