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




// the definition of the exception class
PyObject * mpi::Error = 0;
const char * const mpi::Error__name__ = "Error";

// exception registration
PyObject * mpi::registerExceptionHierarchy(PyObject * module) {

    std::string stem = "mpi.";

    // the base class
    // build its name
    std::string errorName = stem + Error__name__;
    // and the exception object
    Error = PyErr_NewException(errorName.c_str(), 0, 0);
    // increment its reference count so we can pass ownership to the module
    Py_INCREF(Error);
    // register it with the module
    PyModule_AddObject(module, Error__name__, Error);

    // and return the module
    return module;
}

// end of file
