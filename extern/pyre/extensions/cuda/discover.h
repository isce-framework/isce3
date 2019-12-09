// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(pyre_extensions_cuda_discover_h)
#define pyre_extensions_cuda_discover_h


// place everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace cuda {

            // discover
            const char * const discover__name__ = "discover";
            const char * const discover__doc__ = "device discovery";
            PyObject * discover(PyObject *, PyObject *);

        } // of namespace cuda
    } // of namespace extensions
} // of namespace pyre

#endif

// end of file
