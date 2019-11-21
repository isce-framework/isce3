// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#include <portinfo>
#include <Python.h>
#include <pyre/journal.h>

// the module method declarations
#include "connection.h"
#include "exceptions.h"
#include "execute.h"
#include "metadata.h"

// put everything in my private namespace
namespace pyre {
    namespace extensions {
        namespace postgres {

            // the module method table
            PyMethodDef methods[] = {
                // module metadata
                // the copyright method
                { copyright__name__, copyright, METH_VARARGS, copyright__doc__ },
                // the version method
                { version__name__, version, METH_VARARGS, version__doc__ },

                // registration
                { registerExceptions__name__,
                  registerExceptions, METH_VARARGS, registerExceptions__doc__ },
                { registerNULL__name__, registerNULL, METH_VARARGS, registerNULL__doc__ },

                // connections
                { connect__name__, connect, METH_VARARGS, connect__doc__ },
                { disconnect__name__, disconnect, METH_VARARGS, disconnect__doc__ },

                // SQL command execution
                { execute__name__, execute, METH_VARARGS, execute__doc__ },
                { submit__name__, submit, METH_VARARGS, submit__doc__ },
                { busy__name__, busy, METH_VARARGS, busy__doc__ },
                { consume__name__, consume, METH_VARARGS, consume__doc__ },
                { retrieve__name__, retrieve, METH_VARARGS, retrieve__doc__ },

                // sentinel
                {0, 0, 0, 0}
            };


            // the module documentation string
            const char * const doc = "provides access to PostgreSQL databases";

            // the module definition structure
            PyModuleDef module = {
                // header
                PyModuleDef_HEAD_INIT,
                // the name of the module
                "postgres",
                // the module documentation string
                doc,
                // size of the per-interpreter state of the module; -1 if this state is global
                -1,
                // the methods defined in this module
                methods
            };

        } // of namespace postgres
    } // of namespace extensions
} // of namespace pyre


// initialization function for the module
// *must* be called PyInit_postgres
PyMODINIT_FUNC
PyInit_postgres()
{
    // create the module
    PyObject * module = PyModule_Create(&pyre::extensions::postgres::module);

    // create the debug channel
    pyre::journal::debug_t debug("postgres.init");
    debug << pyre::journal::at(__HERE__);

    // check whether module creation succeeded and raise an exception if not
    if (!module) {
        debug << "'postgres' module initialization failed";
    } else {
        debug << "'postgres' module initialization succeeded";
    }

    debug << pyre::journal::endl;

    // and return the newly created module
    return module;
}


// end of file
