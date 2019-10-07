// -*- C++ -*-
//
// michael a.g. aivazis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(pyre_mpi_Error_h__)
#define pyre_mpi_Error_h__


// access to the base class


// forward declarations
namespace pyre {
    namespace mpi {
        class Error;
    }
}

// a wrapper around MPI_Error
class pyre::mpi::Error : public std::exception {

// meta-methods
public:
    inline Error(int code);
    inline ~Error();

// data
private:
    int _code; // the raw MPI error code
};

// get the inline definitions
#define pyre_mpi_Error_icc
#include "Error.icc"
#undef pyre_mpi_Error_icc

#endif

// end of file
