// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(pyre_extensions_cuda_metadata_h)
#define pyre_extensions_cuda_metadata_h


// place everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace cuda {

            // copyright note
            const char * const copyright__name__ = "copyright";
            const char * const copyright__doc__ = "the module copyright string";
            PyObject * copyright(PyObject *, PyObject *);

            // version
            const char * const version__name__ = "version";
            const char * const version__doc__ = "the module version string";
            PyObject * version(PyObject *, PyObject *);

            // license
            const char * const license__name__ = "license";
            const char * const license__doc__ = "the module license string";
            PyObject * license(PyObject *, PyObject *);

        } // of namespace cuda
    } // of namespace extensions
} // of namespace pyre

#endif

// end of file
