// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(pyre_extensions_cuda_device_h)
#define pyre_extensions_cuda_device_h


// place everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace cuda {

            // setDevice
            const char * const setDevice__name__ = "setDevice";
            const char * const setDevice__doc__ = "allocate a device to the current thread";
            PyObject * setDevice(PyObject *, PyObject *);

            // resetDevice
            const char * const resetDevice__name__ = "resetDevice";
            const char * const resetDevice__doc__ = "reset the current device";
            PyObject * resetDevice(PyObject *, PyObject *);

        } // of namespace cuda
    } // of namespace extensions
} // of namespace pyre

#endif

// end of file
