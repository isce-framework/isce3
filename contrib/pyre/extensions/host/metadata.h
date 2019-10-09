// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(pyre_extensions_host_metadata_h)
#define pyre_extensions_host_metadata_h


// place everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace host {

            // copyright note
            const char * const copyright__name__ = "copyright";
            const char * const copyright__doc__ = "the module copyright string";
            PyObject * copyright(PyObject *, PyObject *);

            // version string
            const char * const version__name__ = "version";
            const char * const version__doc__ = "the module version string";
            PyObject * version(PyObject *, PyObject *);

        } // of namespace host
    } // of namespace extensions
} // of namespace pyre

#endif

// end of file
