// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(pyre_extensions_journal_init_h)
#define pyre_extensions_journal_init_h


// place everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace journal {

            // initialization
            const char * const registerJournal__name__ = "registerJournal";
            const char * const registerJournal__doc__ = "the extension initialization";
            PyObject * registerJournal(PyObject *, PyObject *);

        } // of namespace journal
    } // of namespace extensions
} // of namespace pyre

#endif

// end of file
