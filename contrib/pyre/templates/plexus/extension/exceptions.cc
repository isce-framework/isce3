// -*- C++ -*-
//
// {project.authors}
// {project.affiliations}
// (c) {project.span} all rights reserved
//

#include <portinfo>
#include <Python.h>
#include <string>

#include "exceptions.h"

namespace {project.name} {{
    namespace extension {{
        // base class for {project.name} errors
        std::string Error__name__ = "Error";
        PyObject * Error = 0;
    }} // of namespace extension
}} // of namespace {project.name}


// exception registration
PyObject *
{project.name}::extension::
registerExceptionHierarchy(PyObject * module) {{

    std::string stem = "{project.name}.";

    // the base class
    // build its name
    std::string errorName = stem + {project.name}::extension::Error__name__;
    // and the exception object
    {project.name}::extension::Error = PyErr_NewException(errorName.c_str(), 0, 0);
    // increment its reference count so we can pass ownership to the module
    Py_INCREF({project.name}::extension::Error);
    // register it with the module
    PyModule_AddObject(module,
                       {project.name}::extension::Error__name__.c_str(),
                       {project.name}::extension::Error);

    // and return the module
    return module;
}}

// end of file
