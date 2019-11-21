// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#include <portinfo>
#include <Python.h>

// the module method declarations
#include "metadata.h"
#include "cpu.h"

// put everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace host {

            // the module method table
            PyMethodDef methods[] = {
                // module metadata
                // the copyright method
                { copyright__name__, copyright, METH_VARARGS, copyright__doc__ },
                // the version method
                { version__name__, version, METH_VARARGS, version__doc__ },

                // cpu info
                { logical__name__, logical, METH_VARARGS, logical__doc__ },
                { logicalMax__name__, logicalMax, METH_VARARGS, logicalMax__doc__ },
                { physical__name__, physical, METH_VARARGS, physical__doc__ },
                { physicalMax__name__, physicalMax, METH_VARARGS, physicalMax__doc__ },

                // sentinel
                {0, 0, 0, 0}
            };


            // the module documentation string
            const char * const doc = "provides access host specific information";

            // the module definition structure
            PyModuleDef module = {
                // header
                PyModuleDef_HEAD_INIT,
                // the name of the module
                "_host",
                // the module documentation string
                doc,
                // size of the per-interpreter state of the module; -1 if this state is global
                -1,
                // the methods defined in this module
                methods
            };

        } // of namespace host
    } // of namespace extensions
} // of namespace pyre

// initialization function for the module
// *must* be called PyInit_host
PyMODINIT_FUNC
PyInit_host()
{
    // create the module
    PyObject * module = PyModule_Create(&pyre::extensions::host::module);
    // check whether module creation succeeded and raise an exception if not
    if (!module) {
        return module;
    }
    // module initializations
    // and return the newly created module
    return module;
}


// end of file
