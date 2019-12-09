// -*- C++ -*-
// 
// {project.authors}
// {project.affiliations}
// (c) {project.span} all rights reserved
// 

// configuration
#include <portinfo>

// externals
#include <Python.h>
#include <{project.name}/version.h>

// my declarations
#include "metadata.h"


// copyright
const char * const
{project.name}::extension::
copyright__name__ = "copyright";

const char * const
{project.name}::extension::
copyright__doc__ = "the project copyright string";

PyObject * 
{project.name}::extension::
copyright(PyObject *, PyObject *)
{{
    const char * const copyright_note = "{project.name}: (c) {project.span} {project.authors}";
    return Py_BuildValue("s", copyright_note);
}}
    

// version
const char * const
{project.name}::extension::
version__name__ = "version";

const char * const 
{project.name}::extension::
version__doc__ = "the project version string";

PyObject * 
{project.name}::extension::
version(PyObject *, PyObject *)
{{
        return Py_BuildValue("s", {project.name}::version());
}}

    
// end of file
