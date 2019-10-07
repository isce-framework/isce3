// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(pyre_extensions_mpi_exceptions_h)
#define pyre_extensions_mpi_exceptions_h


// place everything in my private namespace
namespace mpi {

    // exception registration
    PyObject * registerExceptionHierarchy(PyObject *);

    // base class for mpi errors
    extern PyObject * Error;
    extern const char * const Error__name__;

} // of namespace mpi

#endif

// end of file
