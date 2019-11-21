// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#include <portinfo>
#include <Python.h>
#include <string>
#include <iostream>
#include <pyre/journal.h>

#include "exceptions.h"

namespace pyre {
    namespace extensions {
        namespace postgres {
            // exception hierarchy for postgres errors
            PyObject * Error = 0;
            PyObject * Warning = 0;
            PyObject * InterfaceError = 0;
            PyObject * DatabaseError = 0;
            PyObject * DataError = 0;
            PyObject * OperationalError = 0;
            PyObject * IntegrityError = 0;
            PyObject * InternalError = 0;
            PyObject * ProgrammingError = 0;
            PyObject * NotSupportedError = 0;
        } // of namespace postgres
    } // of namespace extensions
} // of namespace pyre


// exception registration
const char * const
pyre::extensions::postgres::
registerExceptions__name__ = "registerExceptions";

const char * const
pyre::extensions::postgres::
registerExceptions__doc__ =
    "register the classes that represent the standard exceptions raised by"
    "DB API 2.0 compliant implementations";

PyObject *
pyre::extensions::postgres::
registerExceptions(PyObject * module, PyObject * args) {

    // unpack the arguments
    PyObject * exceptions;
    if (!PyArg_ParseTuple(args, "O!:registerExceptions", &PyModule_Type, &exceptions)) {
        return 0;
    }

    // keep a record of the standard exception classes and register them with the module
    Warning = PyObject_GetAttrString(exceptions, "Warning");
    PyModule_AddObject(module, "Warning", Warning);

    Error = PyObject_GetAttrString(exceptions, "Error");
    PyModule_AddObject(module, "Error", Error);

    InterfaceError = PyObject_GetAttrString(exceptions, "InterfaceError");
    PyModule_AddObject(module, "InterfaceError", InterfaceError);

    DatabaseError = PyObject_GetAttrString(exceptions, "DatabaseError");
    PyModule_AddObject(module, "DatabaseError", DatabaseError);

    DataError = PyObject_GetAttrString(exceptions, "DataError");
    PyModule_AddObject(module, "DataError", DataError);

    OperationalError = PyObject_GetAttrString(exceptions, "OperationalError");
    PyModule_AddObject(module, "OperationalError", OperationalError);

    IntegrityError = PyObject_GetAttrString(exceptions, "IntegrityError");
    PyModule_AddObject(module, "IntegrityError", IntegrityError);

    InternalError = PyObject_GetAttrString(exceptions, "InternalError");
    PyModule_AddObject(module, "InternalError", InternalError);

    ProgrammingError = PyObject_GetAttrString(exceptions, "ProgrammingError");
    PyModule_AddObject(module, "ProgrammingError", ProgrammingError);

    NotSupportedError = PyObject_GetAttrString(exceptions, "NotSupportedError");
    PyModule_AddObject(module, "NotSupportedError", NotSupportedError);

    // and return the module
    Py_INCREF(Py_None);
    return Py_None;
}


// the representation of {NULL} from {pyre.db}; set during extension registration
PyObject * pyre::extensions::postgres::null = 0;


// {NULL} registration
const char * const
pyre::extensions::postgres::
registerNULL__name__ = "registerNULL";

const char * const
pyre::extensions::postgres::
registerNULL__doc__ = "register the representation of the {NULL} object";

PyObject *
pyre::extensions::postgres::
registerNULL(PyObject * module, PyObject * args) {

    // unpack the arguments
    if (!PyArg_ParseTuple(args, "O:registerNULL", &null)) {
        return 0;
    }

    // set up the debug channel
    pyre::journal::debug_t debug("postgres.registration");
    // let me know what the {NULL} rep looks like
    debug
        << pyre::journal::at(__HERE__)
        << "NULL from " << null
        << pyre::journal::endl;

    // and return the module
    Py_INCREF(Py_None);
    return Py_None;
}

// end of file
