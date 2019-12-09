// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(pyre_extensions_journal_exceptions_h)
#define pyre_extensions_journal_exceptions_h


// place everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace journal {

        // exception registration
        PyObject * registerExceptionHierarchy(PyObject *);

        } // of namespace journal
    } // of namespace extensions
} // of namespace pyre

#endif

// end of file
