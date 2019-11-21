// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#include <portinfo>
#include <Python.h>

// the module method declarations
#include "init.h"
#include "exceptions.h"
#include "metadata.h"
#include "channels.h"
#include "tests.h"


// put everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace journal {

            // the module method table
            PyMethodDef module_methods[] = {
                // the copyright method
                { copyright__name__, copyright, METH_VARARGS, copyright__doc__ },
                // the version
                { version__name__, version, METH_VARARGS, version__doc__ },

                // initialization
                { registerJournal__name__, registerJournal, METH_VARARGS, registerJournal__doc__ },

                // channels
                // lookup
                { lookupDebug__name__, lookupDebug, METH_VARARGS, lookupDebug__doc__ },
                { lookupFirewall__name__, lookupFirewall, METH_VARARGS, lookupFirewall__doc__ },
                { lookupInfo__name__, lookupInfo, METH_VARARGS, lookupInfo__doc__ },
                { lookupWarning__name__, lookupWarning, METH_VARARGS, lookupWarning__doc__ },
                { lookupError__name__, lookupError, METH_VARARGS, lookupError__doc__ },

                // access the state of Inventory<true>
                { setEnabledState__name__, setEnabledState, METH_VARARGS, setEnabledState__doc__ },
                { getEnabledState__name__, getEnabledState, METH_VARARGS, getEnabledState__doc__ },
                // access the state of Inventory<false>
                { setDisabledState__name__,
                  setDisabledState, METH_VARARGS, setDisabledState__doc__ },
                { getDisabledState__name__,
                  getDisabledState, METH_VARARGS, getDisabledState__doc__ },

                // examples
                { debugTest__name__, debugTest, METH_VARARGS, debugTest__doc__},
                { firewallTest__name__, firewallTest, METH_VARARGS, firewallTest__doc__},
                { infoTest__name__, infoTest, METH_VARARGS, infoTest__doc__},
                { warningTest__name__, warningTest, METH_VARARGS, warningTest__doc__},
                { errorTest__name__, errorTest, METH_VARARGS, errorTest__doc__},

                // sentinel
                {0, 0, 0, 0}
            };

            // the module documentation string
            const char * const __doc__ = "sample module documentation string";

            // the module definition structure
            PyModuleDef module_definition = {
                // header
                PyModuleDef_HEAD_INIT,
                // the name of the module
                "journal",
                // the module documentation string
                __doc__,
                // size of the per-interpreter state of the module; -1 if this state is global
                -1,
                // the methods defined in this module
                module_methods
            };

        } // of namespace journal
    } // of namespace extensions
} // of namespace pyre


// initialization function for the module
// *must* be called PyInit_journal
PyMODINIT_FUNC
PyInit_journal()
{
    // create the module
    PyObject * module = PyModule_Create(&pyre::extensions::journal::module_definition);
    // check whether module creation succeeded and raise an exception if not
    if (!module) {
        return module;
    }
    // otherwise, we have an initialized module
    pyre::extensions::journal::registerExceptionHierarchy(module);

    // and return the newly created module
    return module;
}

// end of file
