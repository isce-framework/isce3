// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(pyre_extensions_postgres_exceptions_h)
#define pyre_extensions_postgres_exceptions_h


// place everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace postgres {

            // global variables -- ouch
            extern PyObject * null;

            // exception registration
            extern const char * const registerExceptions__name__;
            extern const char * const registerExceptions__doc__;
            PyObject * registerExceptions(PyObject *, PyObject *);

            // registration of the representation of {NULL}
            extern const char * const registerNULL__name__;
            extern const char * const registerNULL__doc__;
            PyObject * registerNULL(PyObject *, PyObject *);

        } // of namespace postgres
    } // of namespace extensions
} // of namespace pyre

#endif

// end of file
