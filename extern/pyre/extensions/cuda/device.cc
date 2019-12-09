// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#include <portinfo>
#include <Python.h>
#include <pyre/journal.h>

// my declarations
#include "device.h"
// local support
#include "exceptions.h"
// so I can build reasonable error messages
#include <sstream>

// access to the CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>


// grab a device
PyObject *
pyre::extensions::cuda::
setDevice(PyObject *, PyObject *args)
{
    // allocate storage for the arguments
    int did;
    // if I were not passed the expected arguments
    if (!PyArg_ParseTuple(args, "i:setDevice", &did)) {
        // raise an exception
        return nullptr;
    }

    // attempt to grab the device
    cudaError_t status = cudaSetDevice(did);
    // if anything went wrong
    if (status != cudaSuccess) {
        // make an error channel
        pyre::journal::error_t error("cuda");
        // show me
        error
            << pyre::journal::at(__HERE__)
            << "while reserving device " << did << ": "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;

        // create an exception object
        // prep the constructor arguments
        PyObject * args = PyTuple_New(0);
        PyObject * kwds = Py_BuildValue("{s:s}", "description", cudaGetErrorName(status));
        // build it
        PyObject * exception = PyObject_Call(Error, args, kwds);
        // mark it as the pending exception
        PyErr_SetObject(Error, exception);
        // and bail
        return nullptr;
    }

    // all done
    Py_INCREF(Py_None);
    return Py_None;
}


// reset a device
PyObject *
pyre::extensions::cuda::
resetDevice(PyObject *, PyObject *args)
{
    // if I were not passed the expected arguments
    if (!PyArg_ParseTuple(args, ":resetDevice", &PyType_Type)) {
        // raise an exception
        return nullptr;
    }

    // attempt to grab the device
    cudaError_t status = cudaDeviceReset();
    // if anything went wrong
    if (status != cudaSuccess) {
        // make an error channel
        pyre::journal::error_t error("cuda");
        // show me
        error
            << pyre::journal::at(__HERE__)
            << "while resetting the current device: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;

        // create an exception object
        // prep the constructor arguments
        PyObject * args = PyTuple_New(0);
        PyObject * kwds = Py_BuildValue("{s:s}", "description", cudaGetErrorName(status));
        // build it
        PyObject * exception = PyObject_Call(Error, args, kwds);
        // mark it as the pending exception
        PyErr_SetObject(Error, exception);
        // and bail
        return nullptr;
    }

    // all done
    Py_INCREF(Py_None);
    return Py_None;
}

// end of file
