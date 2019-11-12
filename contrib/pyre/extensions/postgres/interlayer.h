// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(pyre_extensions_postgres_interlayer_h)
#define pyre_extensions_postgres_interlayer_h


// place everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace postgres {

            // types
            typedef const char * const string_t;
            typedef PyObject * (*resultProcessor_t)(PGresult *);

            // other utilities
            PyObject * buildResultTuple(PGresult *);

            PyObject * processResult(
                                     string_t command,
                                     PGresult * result,
                                     resultProcessor_t processor);

            // exceptions
            PyObject * raiseOperationalError(string_t description);
            PyObject * raiseProgrammingError(string_t description, string_t command);

        } // of namespace postgres
    } // of namespace extensions
} // of namespace pyre

#endif

// end of file
