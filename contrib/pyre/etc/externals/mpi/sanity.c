// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// get the headers
#include <mpi.h>


// smallest possible driver
int main() {
    // initialize MPI
    MPI_Init(0,0);
    // finalize MPI
    MPI_Finalize();

    // all done
    return 0;
}


// end of file
