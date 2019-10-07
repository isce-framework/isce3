// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(gsl_extension_exceptions_h)
#define gsl_extension_exceptions_h


// place everything in my private namespace
namespace gsl {
    // base class for gsl errors
    extern PyObject * Error;
    extern const char * const Error__name__;

} // of namespace gsl

#endif

// end of file
