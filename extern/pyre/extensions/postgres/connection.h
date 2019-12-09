// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(pyre_extensions_postgres_connection_h)
#define pyre_extensions_postgres_connection_h

namespace pyre {
    namespace extensions {
        namespace postgres {

            // establish a connection to the pg back end
            extern const char * const connect__name__;
            extern const char * const connect__doc__;
            PyObject * connect(PyObject *, PyObject *);

            // disconnect from the back end
            extern const char * const disconnect__name__;
            extern const char * const disconnect__doc__;
            PyObject * disconnect(PyObject *, PyObject *);

        } // of namespace postgres
    } // of namespace extensions
} // of namespace pyre


# endif

// end of file
