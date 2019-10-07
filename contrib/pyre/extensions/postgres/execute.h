// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(pyre_extensions_postgres_execute_h)
#define pyre_extensions_postgres_execute_h

namespace pyre {
    namespace extensions {
        namespace postgres {

            // establish a connection to the pg back end
            extern const char * const execute__name__;
            extern const char * const execute__doc__;
            PyObject * execute(PyObject *, PyObject *);

            // submit a query for asynchronous processing
            extern const char * const submit__name__;
            extern const char * const submit__doc__;
            PyObject * submit(PyObject *, PyObject *);

            // consume partial results from the server
            extern const char * const consume__name__;
            extern const char * const consume__doc__;
            PyObject * consume(PyObject *, PyObject *);

            // retrieve results from the server
            extern const char * const retrieve__name__;
            extern const char * const retrieve__doc__;
            PyObject * retrieve(PyObject *, PyObject *);

            // check whether a result set is available
            extern const char * const busy__name__;
            extern const char * const busy__doc__;
            PyObject * busy(PyObject *, PyObject *);

        } // of namespace postgres
    } // of namespace extensions
} // of namespace pyre

# endif

// end of file
