// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#include <portinfo>
#include <Python.h>
#include <string>

#include "exceptions.h"

namespace pyre {
    namespace extensions {
        namespace journal {
        // base class for journal errors
        const char * const Error__name__ = "Error";
        PyObject * Error = 0;

        } // of namespace journal
    } // of namespace extensions
} // of namespace pyre


// exception registration
PyObject *
pyre::extensions::journal::
registerExceptionHierarchy(PyObject * module) {

    std::string stem = "journal.";

    // the base class
    // build its name
    std::string errorName = stem + journal::Error__name__;
    // and the exception object
    journal::Error = PyErr_NewException(errorName.c_str(), 0, 0);
    // increment its reference count so we can pass ownership to the module
    Py_INCREF(journal::Error);
    // register it with the module
    PyModule_AddObject(module, journal::Error__name__, journal::Error);

    // and return the module
    return module;
}

// end of file
