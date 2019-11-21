// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#include <portinfo>
#include <Python.h>

#include "metadata.h"


// copyright
const char * const
pyre::extensions::journal::
copyright__name__ = "copyright";

const char * const
pyre::extensions::journal::
copyright__doc__ = "the module copyright string";

PyObject *
pyre::extensions::journal::
copyright(PyObject *, PyObject *)
{
    const char * const copyright_note = "journal: (c) 1998-2019 Michael A.G. Aïvázis";
    return Py_BuildValue("s", copyright_note);
}


// version
const char * const
pyre::extensions::journal::
version__name__ = "version";

const char * const
pyre::extensions::journal::
version__doc__ = "the module version string";

PyObject *
pyre::extensions::journal::
version(PyObject *, PyObject *)
{
    const char * const version_string = "0.0";
    return Py_BuildValue("s", version_string);
}


// end of file
