// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// externals
#include <portinfo>
#include <Python.h>
#include <pyre/timers.h>

// access the declarations
#include "display.h"

namespace pyre {
    namespace extensions {
        namespace timers {

            // the capsule tag
            const char * const timerCapsuleName = "pyre.timers.timer";

            // local alias for the timer type
            using timer_t = pyre::timer_t::timer_t;
        } // of namespace timers
    } // of namespace extensions
} // of namespace pyre


// newTimer
PyObject * pyre::extensions::timers::newTimer(PyObject *, PyObject * args)
{
    // the name of the timer
    const char * name;
    // extract the arguments
    if (!PyArg_ParseTuple(args, "s:newTimer", &name)) {
        // and bail if something went wrong
        return nullptr;
    }

    // access the timer
    timer_t * timer = & pyre::timer_t::retrieveTimer(name);

    // encapsulate it
    PyObject * capsule = PyCapsule_New(timer, timerCapsuleName, nullptr);
    // and return the capsule
    return capsule;
}

// start
PyObject * pyre::extensions::timers::start(PyObject *, PyObject * args)
{
    // the capsule with the timer pointer
    PyObject * capsule;
    // extract the arguments
    if (!PyArg_ParseTuple(args, "O!:start", &PyCapsule_Type, &capsule)) {
        // and bail if something went wrong
        return nullptr;
    }

    // if the capsule is invalid
    if (!PyCapsule_IsValid(capsule, timerCapsuleName)) {
        // bail
        return nullptr;
    }

    // cast it to a {timer_t}
    timer_t * timer =
        reinterpret_cast<timer_t *>(PyCapsule_GetPointer(capsule, timerCapsuleName));

    // start the timer
    timer->start();

    // all done
    Py_INCREF(Py_None);
    return Py_None;
}

// stop
PyObject * pyre::extensions::timers::stop(PyObject *, PyObject * args)
{
    // the capsule with the timer pointer
    PyObject * capsule;
    // extract the arguments
    if (!PyArg_ParseTuple(args, "O!:start", &PyCapsule_Type, &capsule)) {
        return nullptr;
    }

    // if the capsule is invalid
    if (!PyCapsule_IsValid(capsule, timerCapsuleName)) {
        // bail
        return nullptr;
    }

    // cast it to a {timer_t}
    timer_t * timer
        = reinterpret_cast<timer_t *>(PyCapsule_GetPointer(capsule, timerCapsuleName));

    // stop the timer
    timer->stop();

    // all done
    Py_INCREF(Py_None);
    return Py_None;
}

// reset
PyObject * pyre::extensions::timers::reset(PyObject *, PyObject * args)
{
    // the capsule with the timer pointer
    PyObject * capsule;
    // extract the arguments
    if (!PyArg_ParseTuple(args, "O!:start", &PyCapsule_Type, &capsule)) {
        // and bail if something went wrong
        return nullptr;
    }

    // if the capsule is invalid
    if (!PyCapsule_IsValid(capsule, timerCapsuleName)) {
        // bail
        return nullptr;
    }
    // cast it to a {timer_t}
    timer_t * timer
        = reinterpret_cast<timer_t *>(PyCapsule_GetPointer(capsule, timerCapsuleName));

    // reset the timer
    timer->reset();

    // all done
    Py_INCREF(Py_None);
    return Py_None;
}

// read
PyObject * pyre::extensions::timers::read(PyObject *, PyObject * args)
{
    // the capsule with the timer pointer
    PyObject * capsule;
    // extract the arguments
    if (!PyArg_ParseTuple(args, "O!:start", &PyCapsule_Type, &capsule)) {
        // if something went wrong, bail
        return nullptr;
    }

    // if the capsule is invalid
    if (!PyCapsule_IsValid(capsule, timerCapsuleName)) {
        // bail
        return nullptr;
    }
    // cast it to a Ptimer_t}
    timer_t * timer =
        reinterpret_cast<timer_t *>(PyCapsule_GetPointer(capsule, timerCapsuleName));

    // read the timer
    double elapsed = timer->read();

    // and return the time elapsed
    return Py_BuildValue("d", elapsed);
}

// lap
PyObject * pyre::extensions::timers::lap(PyObject *, PyObject * args)
{
    // the capsule with the timer pointer
    PyObject * capsule;
    // extract the arguments
    if (!PyArg_ParseTuple(args, "O!:start", &PyCapsule_Type, &capsule)) {
        // if something went wrong, bail
        return nullptr;
    }

    // if the capsule is invalid
    if (!PyCapsule_IsValid(capsule, timerCapsuleName)) {
        // bail
        return nullptr;
    }

    // cast it to a Timer pointer
    timer_t * timer
        = reinterpret_cast<timer_t *>(PyCapsule_GetPointer(capsule, timerCapsuleName));

    // compute the elapsed time
    double elapsed = timer->lap();

    // and return it
    return Py_BuildValue("d", elapsed);
}

// end of file
