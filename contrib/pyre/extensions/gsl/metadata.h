// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(gsl_extension_metadata_h)
#define gsl_extension_metadata_h


// place everything in my private namespace
namespace gsl {

    // copyright note
    extern const char * const copyright__name__;
    extern const char * const copyright__doc__;
    PyObject * copyright(PyObject *, PyObject *);

    // license
    extern const char * const license__name__;
    extern const char * const license__doc__;
    PyObject * license(PyObject *, PyObject *);

    // version
    extern const char * const version__name__;
    extern const char * const version__doc__;
    PyObject * version(PyObject *, PyObject *);

} // of namespace gsl

#endif

// end of file
