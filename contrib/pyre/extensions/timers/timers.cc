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
#include "display.h"

// put everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace timers {

            // the module method table
            PyMethodDef methods[] = {
                // module metadata
                // the copyright method
                { copyright__name__, copyright, METH_VARARGS, copyright__doc__ },
                // the version method
                { version__name__, version, METH_VARARGS, version__doc__ },

                // timer access
                { newTimer__name__, newTimer, METH_VARARGS, newTimer__doc__ },
                { start__name__, start, METH_VARARGS, start__doc__ },
                { stop__name__, stop, METH_VARARGS, stop__doc__ },
                { reset__name__, reset, METH_VARARGS, reset__doc__ },
                { lap__name__, lap, METH_VARARGS, lap__doc__ },
                { read__name__, read, METH_VARARGS, read__doc__ },

                // sentinel
                {0, 0, 0, 0}
            };


            // the module documentation string
            const char * const doc = "provides access to the high resolution pyre timers";

            // the module definition structure
            PyModuleDef module = {
                // header
                PyModuleDef_HEAD_INIT,
                // the name of the module
                "_timers",
                // the module documentation string
                doc,
                // size of the per-interpreter state of the module; -1 if this state is global
                -1,
                // the methods defined in this module
                methods
            };

        } // of namespace timers
    } // of namespace extensions
} // of namespace pyre

// initialization function for the module
// *must* be called PyInit_timers
PyMODINIT_FUNC
PyInit_timers()
{
    // create the module
    PyObject * module = PyModule_Create(&pyre::extensions::timers::module);
    // check whether module creation succeeded and raise an exception if not
    if (!module) {
        return module;
    }
    // module initializations
    // and return the newly created module
    return module;
}


// end of file
