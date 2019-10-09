// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// externals
#include <portinfo>
#include <Python.h>

// access the declarations
#include "cpu.h"

// sysctl
#include <sys/sysctl.h>

// logical
PyObject * pyre::extensions::host::logical(PyObject *, PyObject * args)
{
    // extract the arguments
    if (!PyArg_ParseTuple(args, ":logical")) {
        return 0;
    }

    // storage for the cpu count
    int cpus = 0;

#if defined(HAVE_SYSCTL_HW_DOT)
    // the mib vector and its size
    int mib[2];
    size_t mib_l = sizeof(mib) / sizeof(int);
    // initialize it
    sysctlnametomib("hw.logicalcpu", mib, &mib_l);

    // storage size for the cpu count
    size_t cpus_l = sizeof(cpus);
    // find out how many
    sysctl(mib, mib_l, &cpus, &cpus_l, 0, 0);
#endif

    // return
    return PyLong_FromLong(cpus);
}

// logicalMax
PyObject * pyre::extensions::host::logicalMax(PyObject *, PyObject * args)
{
    // extract the arguments
    if (!PyArg_ParseTuple(args, ":logicalMax")) {
        return 0;
    }

    // storage for the cpu count
    int cpus = 0;

#if defined(HAVE_SYSCTL_HW_DOT)
    // the mib vector and its size
    int mib[2];
    size_t mib_l = sizeof(mib) / sizeof(int);
    // initialize it
    sysctlnametomib("hw.logicalcpu_max", mib, &mib_l);

    // storage size for the cpu count
    size_t cpus_l = sizeof(cpus);
    // find out how many
    sysctl(mib, mib_l, &cpus, &cpus_l, 0, 0);
#endif

    // return
    return PyLong_FromLong(cpus);
}

// physical
PyObject * pyre::extensions::host::physical(PyObject *, PyObject * args)
{
    // extract the arguments
    if (!PyArg_ParseTuple(args, ":physical")) {
        return 0;
    }

    // storage for the cpu count
    int cpus = 0;

#if defined(HAVE_SYSCTL_HW_DOT)
    // the mib vector and its size
    int mib[2];
    size_t mib_l = sizeof(mib) / sizeof(int);
    // initialize it
    sysctlnametomib("hw.physicalcpu", mib, &mib_l);

    // storage size for the cpu count
    size_t cpus_l = sizeof(cpus);
    // find out how many
    sysctl(mib, mib_l, &cpus, &cpus_l, 0, 0);
#endif

    // return
    return PyLong_FromLong(cpus);
}

// physicalMax
PyObject * pyre::extensions::host::physicalMax(PyObject *, PyObject * args)
{
    // extract the arguments
    if (!PyArg_ParseTuple(args, ":physicalMax")) {
        return 0;
    }

    // storage for the cpu count
    int cpus = 0;

#if defined(HAVE_SYSCTL_HW_DOT)
    // the mib vector and its size
    int mib[2];
    size_t mib_l = sizeof(mib) / sizeof(int);
    // initialize it
    sysctlnametomib("hw.physicalcpu_max", mib, &mib_l);

    // storage size for the cpu count
    size_t cpus_l = sizeof(cpus);
    // find out how many
    sysctl(mib, mib_l, &cpus, &cpus_l, 0, 0);
#endif

    // return
    return PyLong_FromLong(cpus);
}

// end of file
