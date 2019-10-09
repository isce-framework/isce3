// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//


// for the build system
#include <portinfo>

// external packages
#include <Python.h>
#include <pyre/journal.h>

// my class header
#include "DeviceProxy.h"


// interface
void
pyre::extensions::journal::DeviceProxy::
record(entry_t & entry, metadata_t & metadata)
{
    // ask the device owner for the current value of the python device
    PyObject * device = PyObject_GetAttrString(_keeper, "device");
    // if {journal.device} failed
    if (!device) {
        // not much we can do...
        return;
    }
    // build a python string for the {record} method
    PyObject * record = PyObject_GetAttrString(device, "record");
    // if {journal.device.record} failed to produce the associated method
    if (!record) {
        // not much we can do...
        Py_DECREF(device);
        return;
    }

    // build a tuple with the strings from {entry}
    // retrieve the number of lines in {entry}
    size_t lines = entry.size();
    // create a tuple to hold the lines in {entry}
    PyObject * page = PyTuple_New(lines);
    // loop over the strings in entry and place them in the tuple
    for (size_t i = 0; i < lines; ++i) {
        // build a python string
        PyObject * line = PyUnicode_FromString(entry[i].c_str());
        // place it in the tuple
        PyTuple_SET_ITEM(page, i, line);
    }

    // build a dictionary with the metadata
    PyObject * meta = PyDict_New();
    // loop over the contents of {metadata}
    for (metadata_t::const_iterator i = metadata.begin(); i != metadata.end(); ++i) {
        // extract the key and value
        string_t key = i->first;
        string_t value = i->second;
        // turn the key into a python string
        PyObject * pykey = PyUnicode_FromString(key.c_str());
        // turn the value into a python string
        PyObject * pyvalue = PyUnicode_FromString(value.c_str());
        // add them to the dictionary
        PyDict_SetItem(meta, pykey, pyvalue);
        // clean up
        Py_DECREF(pykey);
        Py_DECREF(pyvalue);
    }

    // invoke its {record} method with our {page} and {meta} as arguments
    PyObject * result = PyObject_CallFunctionObjArgs(record, page, meta, 0);

    // clean up
    Py_DECREF(page);
    Py_DECREF(meta);
    Py_DECREF(device);
    Py_DECREF(record);

    // if {journal.device.record(...)} succeeded
    if (result) {
        // clean up {result} as well
        Py_DECREF(result);
    }

    // all done
    return;
}


// destructor
pyre::extensions::journal::DeviceProxy::
~DeviceProxy()
{
    Py_DECREF(_keeper);
}


// end of file
