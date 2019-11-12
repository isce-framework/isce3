// -*- C++ -*-
// 
// {project.authors}
// {project.affiliations}
// (c) {project.span} all rights reserved
// 

#include <portinfo>
#include <Python.h>

// the module method declarations
#include "exceptions.h"
#include "metadata.h"


// put everything in my private namespace
namespace {project.name} {{
    namespace extension {{

        // the module method table
        PyMethodDef module_methods[] = {{
            // the copyright method
            {{ copyright__name__, copyright, METH_VARARGS, copyright__doc__ }},
            // the version
            {{ version__name__, version, METH_VARARGS, version__doc__ }},
    
            // sentinel
            {{ 0, 0, 0, 0 }}
        }};

        // the module documentation string
        const char * const __doc__ = "sample project documentation string";

        // the module definition structure
        PyModuleDef module_definition = {{
            // header
            PyModuleDef_HEAD_INIT,
            // the name of the module
            "{project.name}", 
            // the module documentation string
            __doc__,
            // size of the per-interpreter state of the module; -1 if this state is global
            -1,
            // the methods defined in this module
            module_methods
        }};

    }} // of namespace extension
}} // of namespace {project.name}


// initialization function for the module
// *must* be called PyInit_{project.name}
PyMODINIT_FUNC
PyInit_{project.name}()
{{
    // create the module
    PyObject * module = PyModule_Create(&{project.name}::extension::module_definition);
    // check whether module creation succeeded and raise an exception if not
    if (!module) {{
        return module;
    }}
    // otherwise, we have an initialized module
    {project.name}::extension::registerExceptionHierarchy(module);

    // and return the newly created module
    return module;
}}

// end of file
