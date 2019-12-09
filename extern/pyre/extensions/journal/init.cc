// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#include <portinfo>
#include <iostream>
#include <Python.h>
#include <pyre/journal.h>

#include "init.h"
#include "DeviceProxy.h"

using namespace pyre::extensions::journal;


// initialize
PyObject *
pyre::extensions::journal::
registerJournal(PyObject *, PyObject * args)
{
    // accept one argument
    PyObject * journal; // the class that keeps a reference to the default device
    // extract it from the argument tuple
    if (!PyArg_ParseTuple(args, "O:registerJournal", &journal)) {
        return 0;
    }

    // build a new device handler
    DeviceProxy * device = new DeviceProxy(journal);

    // attach it as the default device
    pyre::journal::Chronicler::defaultDevice(device);

    // and return
    Py_INCREF(Py_None);
    return Py_None;
}


// end of file
