// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#include <portinfo>
#include <Python.h>
#include <pyre/journal.h>

#include "tests.h"

// debug
const char * const
pyre::extensions::journal::
debugTest__name__ = "debugTest";

const char * const
pyre::extensions::journal::
debugTest__doc__ = "test the debug channel";

PyObject *
pyre::extensions::journal::
debugTest(PyObject *, PyObject * args)
{
    // storage for the name of the channel
    const char * name;
    // extract the arguments
    if (!PyArg_ParseTuple(args, "s:debugTest", &name)) {
        return 0;
    }

    // build the channel
    pyre::journal::debug_t debug(name);

    // say something
    debug
        << pyre::journal::at(__HERE__)
        << "here is a debug message from C++"
        << pyre::journal::endl;

    // and return
    Py_INCREF(Py_None);
    return Py_None;
}


// firewall
const char * const
pyre::extensions::journal::
firewallTest__name__ = "firewallTest";

const char * const
pyre::extensions::journal::
firewallTest__doc__ = "test the firewall channel";

PyObject *
pyre::extensions::journal::
firewallTest(PyObject *, PyObject * args)
{
    // storage for the name of the channel
    const char * name;
    // extract the arguments
    if (!PyArg_ParseTuple(args, "s:firewallTest", &name)) {
        return 0;
    }

    // build the channel
    pyre::journal::firewall_t firewall(name);

    // say something
    firewall
        << pyre::journal::at(__HERE__)
        << "here is a firewall from C++"
        << pyre::journal::endl;

    // and return
    Py_INCREF(Py_None);
    return Py_None;
}


// info
const char * const
pyre::extensions::journal::
infoTest__name__ = "infoTest";

const char * const
pyre::extensions::journal::
infoTest__doc__ = "test the info channel";

PyObject *
pyre::extensions::journal::
infoTest(PyObject *, PyObject * args)
{
    // storage for the name of the channel
    const char * name;
    // extract the arguments
    if (!PyArg_ParseTuple(args, "s:infoTest", &name)) {
        return 0;
    }

    // build the channel
    pyre::journal::info_t info(name);

    // say something
    info
        << pyre::journal::at(__HERE__)
        << "here is an informational from C++"
        << pyre::journal::endl;

    // and return
    Py_INCREF(Py_None);
    return Py_None;
}


// warning
const char * const
pyre::extensions::journal::
warningTest__name__ = "warningTest";

const char * const
pyre::extensions::journal::
warningTest__doc__ = "test the warning channel";

PyObject *
pyre::extensions::journal::
warningTest(PyObject *, PyObject * args)
{
    // storage for the name of the channel
    const char * name;
    // extract the arguments
    if (!PyArg_ParseTuple(args, "s:warningTest", &name)) {
        return 0;
    }

    // build the channel
    pyre::journal::warning_t warning(name);

    // say something
    warning
        << pyre::journal::at(__HERE__)
        << "here is a warning from C++"
        << pyre::journal::endl;

    // and return
    Py_INCREF(Py_None);
    return Py_None;
}


// error
const char * const
pyre::extensions::journal::
errorTest__name__ = "errorTest";

const char * const
pyre::extensions::journal::
errorTest__doc__ = "test the error channel";

PyObject *
pyre::extensions::journal::
errorTest(PyObject *, PyObject * args)
{
    // storage for the name of the channel
    const char * name;
    // extract the arguments
    if (!PyArg_ParseTuple(args, "s:errorTest", &name)) {
        return 0;
    }

    // build the channel
    pyre::journal::error_t error(name);

    // say something
    error
        << pyre::journal::at(__HERE__)
        << "here is an error from C++"
        << pyre::journal::endl;

    // and return
    Py_INCREF(Py_None);
    return Py_None;
}


// end of file
