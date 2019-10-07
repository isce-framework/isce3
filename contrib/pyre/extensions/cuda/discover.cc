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
#include "discover.h"
// local support
#include "exceptions.h"
// so I can build reasonable error messages
#include <sstream>

// access to the CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>

// helpers
inline int coresPerProcessor(int major, int minor);

// device discovery
PyObject *
pyre::extensions::cuda::
discover(PyObject *, PyObject *args)
{
    // the device property class; it's supposed to be a class, so it's an instance of {type}
    PyObject *sheetFactory;
    // my journal channel; for debugging
    pyre::journal::debug_t channel("cuda");

    // if I were not passed the expected arguments
    if (!PyArg_ParseTuple(args, "O!:discover", &PyType_Type, &sheetFactory)) {
        // raise an exception
        return 0;
    }

    // find out how many devices there are
    int count;
    cudaError_t status = cudaGetDeviceCount(&count);
    // if anything went wrong
    if (status != cudaSuccess) {
        // make an error channel
        pyre::journal::error_t error("cuda");
        // show me
        error
            << pyre::journal::at(__HERE__)
            << "while getting device count: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // pretend there are no CUDA capable devices
        return PyTuple_New(0);
    }
    // show me
    channel << "CUDA devices: " << count << pyre::journal::endl;

    // build the device tuple
    PyObject * result = PyTuple_New(count);
    // if there are no devices attached
    if (!count) {
        // why are we here?
        return result;
    }

    // loop over the available devices
    for (int index=0; index<count; ++index) {
        // make a device property sheet
        PyObject *sheet = PyObject_CallObject(sheetFactory, 0);
        // add it to our pile
        PyTuple_SET_ITEM(result, index, sheet);

        // start decorating: first the device id
        PyObject_SetAttrString(sheet, "id", PyLong_FromLong(index));

        // storage for the device properties
        cudaDeviceProp prop;
        // set the current device
        cudaSetDevice(index);
        // get its properties
        cudaGetDeviceProperties(&prop, index);

        // get the name of the device
        PyObject_SetAttrString(sheet, "name", PyUnicode_FromString(prop.name));

        // build a representation of the compute capability
        PyObject * capability = PyTuple_New(2);
        PyTuple_SET_ITEM(capability, 0, PyLong_FromLong(prop.major));
        PyTuple_SET_ITEM(capability, 1, PyLong_FromLong(prop.minor));
        // attach it
        PyObject_SetAttrString(sheet, "capability", capability);

        // version info
        int version;
        PyObject *vtuple;
        // get the driver version
        cudaDriverGetVersion(&version);
        // build a rep for the driver version
        vtuple = PyTuple_New(2);
        PyTuple_SET_ITEM(vtuple, 0, PyLong_FromLong(version/1000));
        PyTuple_SET_ITEM(vtuple, 1, PyLong_FromLong((version%100)/10));
        // attach it
        PyObject_SetAttrString(sheet, "driverVersion", vtuple);

        // get the runtime version
        cudaRuntimeGetVersion(&version);
        // build a rep for the runtime version
        vtuple = PyTuple_New(2);
        PyTuple_SET_ITEM(vtuple, 0, PyLong_FromLong(version/1000));
        PyTuple_SET_ITEM(vtuple, 1, PyLong_FromLong((version%100)/10));
        // attach it
        PyObject_SetAttrString(sheet, "runtimeVersion", vtuple);

        // attach the compute mode
        PyObject_SetAttrString(sheet, "computeMode", PyLong_FromLong(prop.computeMode));

        // attach the managed memory flag
        PyObject_SetAttrString(sheet,
                               "managedMemory",
                               PyBool_FromLong(prop.managedMemory));
        // attach the unified addressing flag
        PyObject_SetAttrString(sheet,
                               "unifiedAddressing",
                               PyBool_FromLong(prop.unifiedAddressing));

        // get the number of multiprocessors
        int processors = prop.multiProcessorCount;
        // attach
        PyObject_SetAttrString(sheet, "processors", PyLong_FromLong(processors));
        // get number of cores per multiprocessor
        int cores = coresPerProcessor(prop.major, prop.minor);
        // attach
        PyObject_SetAttrString(sheet, "coresPerProcessor", PyLong_FromLong(cores));

        // total global memory
        PyObject_SetAttrString(sheet,
                               "globalMemory",
                               PyLong_FromUnsignedLong(prop.totalGlobalMem));
        // total constant memory
        PyObject_SetAttrString(sheet,
                               "constantMemory",
                               PyLong_FromUnsignedLong(prop.totalConstMem));
        // shared memory per block
        PyObject_SetAttrString(sheet,
                               "sharedMemoryPerBlock",
                               PyLong_FromUnsignedLong(prop.sharedMemPerBlock));

        // warp size
        PyObject_SetAttrString(sheet,
                               "warp",
                               PyLong_FromLong(prop.warpSize));
        // maximum number of threads per block
        PyObject_SetAttrString(sheet,
                               "maxThreadsPerBlock",
                               PyLong_FromLong(prop.maxThreadsPerBlock));
        // maximum number of threads per processor
        PyObject_SetAttrString(sheet,
                               "maxThreadsPerProcessor",
                               PyLong_FromLong(prop.maxThreadsPerMultiProcessor));

        // build a rep for the max grid dimensions
        vtuple = PyTuple_New(3);
        // populate it
        PyTuple_SET_ITEM(vtuple, 0, PyLong_FromLong(prop.maxGridSize[0]));
        PyTuple_SET_ITEM(vtuple, 1, PyLong_FromLong(prop.maxGridSize[1]));
        PyTuple_SET_ITEM(vtuple, 2, PyLong_FromLong(prop.maxGridSize[2]));
        // attach it
        PyObject_SetAttrString(sheet, "maxGrid", vtuple);

        // build a rep for the max thread block dimensions
        vtuple = PyTuple_New(3);
        // populate it
        PyTuple_SET_ITEM(vtuple, 0, PyLong_FromLong(prop.maxThreadsDim[0]));
        PyTuple_SET_ITEM(vtuple, 1, PyLong_FromLong(prop.maxThreadsDim[1]));
        PyTuple_SET_ITEM(vtuple, 2, PyLong_FromLong(prop.maxThreadsDim[2]));
        // attach it
        PyObject_SetAttrString(sheet, "maxThreadBlock", vtuple);
    }

    // return the device tuple
    return result;
}


// helpers
// the layout of each row of the architecture table
struct coreTableEntry {
    int version;
    int cores;
};

// the known GPU generations
static coreTableEntry coreTableMap[] = {
    { 0x70,  64}, // Volta Generation (SM 7.0) GV100 class
    { 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
    { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
    { 0x60,  64}, // Pascal Generation (SM 6.0) GP100 class
    { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
    { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
    { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
    { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
    { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
    { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
    { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
};

// the number of known GPU generations
static const int nGenerations = sizeof(coreTableMap)/sizeof(coreTableEntry);

// scan through the table looking for the corresponding number of cores
inline int coresPerProcessor(int major, int minor) {
    // go through the table
    for (std::size_t index=0; index < nGenerations; ++index) {
        // check the encoded version number
        if (coreTableMap[index].version == ((major<<4) + minor)) {
            // if there is a match, look up the number of cores and return it
            return coreTableMap[index].cores;
        }
    }

    // if there is no match, the table is out of date
    // create a firewall
    pyre::journal::firewall_t channel("cuda");
    // complain
    channel
        << pyre::journal::at(__HERE__)
        << "core count for generation (" << major << "," << minor << ") is unknown"
        << pyre::journal::endl;

    // return junk
    return 0;
}


// end of file
