// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#include <portinfo>
#include <Python.h>
#include <pyre/journal.h>

#include "channels.h"

// typedefs
typedef pyre::journal::Inventory<true> enabled_t;
typedef pyre::journal::Inventory<false> disabled_t;

// capsule names
static const char * enabledInventoryCapsuleName = "enabledInventory";
static const char * disabledInventoryCapsuleName = "disabledInventory";

// lookupDebug
const char * const
pyre::extensions::journal::
lookupDebug__name__ = "lookupDebugInventory";

const char * const
pyre::extensions::journal::
lookupDebug__doc__ = "get the channel state from the debug index";

PyObject *
pyre::extensions::journal::
lookupDebug(PyObject *, PyObject * args)
{
    // storage for the name of the channel
    const char * name;
    // extract the arguments
    if (!PyArg_ParseTuple(args, "s:lookupDebugInventory", &name)) {
        return 0;
    }
    // access the state
    disabled_t * inventory = &pyre::journal::debug_t::lookup(name);
    // encapsulate it and return it
    return PyCapsule_New(inventory, disabledInventoryCapsuleName, 0);
}


// lookupFirewall
const char * const
pyre::extensions::journal::
lookupFirewall__name__ = "lookupFirewallInventory";

const char * const
pyre::extensions::journal::
lookupFirewall__doc__ = "get the channel state from the firewall index";

PyObject *
pyre::extensions::journal::
lookupFirewall(PyObject *, PyObject * args)
{
    // storage for the name of the channel
    const char * name;
    // extract the arguments
    if (!PyArg_ParseTuple(args, "s:lookupFirewallInventory", &name)) {
        return 0;
    }
    // access the state
    enabled_t * inventory = &pyre::journal::firewall_t::lookup(name);
    // encapsulate it and return it
    return PyCapsule_New(inventory, enabledInventoryCapsuleName, 0);
}


// lookupInfo
const char * const
pyre::extensions::journal::
lookupInfo__name__ = "lookupInfoInventory";

const char * const
pyre::extensions::journal::
lookupInfo__doc__ = "get the channel state from the info index";

PyObject *
pyre::extensions::journal::
lookupInfo(PyObject *, PyObject * args)
{
    // storage for the name of the channel
    const char * name;
    // extract the arguments
    if (!PyArg_ParseTuple(args, "s:lookupInfoInventory", &name)) {
        return 0;
    }
    // access the state
    enabled_t * inventory = &pyre::journal::info_t::lookup(name);
    // encapsulate it and return it
    return PyCapsule_New(inventory, enabledInventoryCapsuleName, 0);
}


// lookupWarning
const char * const
pyre::extensions::journal::
lookupWarning__name__ = "lookupWarningInventory";

const char * const
pyre::extensions::journal::
lookupWarning__doc__ = "get the channel state from the warning index";

PyObject *
pyre::extensions::journal::
lookupWarning(PyObject *, PyObject * args)
{
    // storage for the name of the channel
    const char * name;
    // extract the arguments
    if (!PyArg_ParseTuple(args, "s:lookupWarningInventory", &name)) {
        return 0;
    }
    // access the state
    enabled_t * inventory = &pyre::journal::warning_t::lookup(name);
    // encapsulate it and return it
    return PyCapsule_New(inventory, enabledInventoryCapsuleName, 0);
}


// lookupError
const char * const
pyre::extensions::journal::
lookupError__name__ = "lookupErrorInventory";

const char * const
pyre::extensions::journal::
lookupError__doc__ = "get the channel state from the error index";

PyObject *
pyre::extensions::journal::
lookupError(PyObject *, PyObject * args)
{
    // storage for the name of the channel
    const char * name;
    // extract the arguments
    if (!PyArg_ParseTuple(args, "s:lookupErrorInventory", &name)) {
        return 0;
    }
    // access the state
    enabled_t * inventory = &pyre::journal::error_t::lookup(name);
    // encapsulate it and return it
    return PyCapsule_New(inventory, enabledInventoryCapsuleName, 0);
}


// setEnabledState
const char * const
pyre::extensions::journal::
setEnabledState__name__ = "setEnabledState";

const char * const
pyre::extensions::journal::
setEnabledState__doc__ = "set the state of a normally enabled channel";

PyObject *
pyre::extensions::journal::
setEnabledState(PyObject *, PyObject * args)
{
    // accept two parameters
    PyObject * state;
    PyObject * inventoryCapsule;
    // extract them
    if (!PyArg_ParseTuple(
                          args, "O!O!",
                          &PyCapsule_Type, &inventoryCapsule,
                          &PyBool_Type, &state)) {
        return 0;
    }
    // decapsulate the inventory
    enabled_t * inventory =
        static_cast<enabled_t *>
        (PyCapsule_GetPointer(inventoryCapsule, enabledInventoryCapsuleName));
    // adjust the state
    if (state == Py_True) {
        inventory->activate();
    } else {
        inventory->deactivate();
    }

    Py_INCREF(Py_None);
    return Py_None;
}


// getEnabledState
const char * const
pyre::extensions::journal::
getEnabledState__name__ = "getEnabledState";

const char * const
pyre::extensions::journal::
getEnabledState__doc__ = "get the state of a normally enabled channel";

PyObject *
pyre::extensions::journal::
getEnabledState(PyObject *, PyObject * args)
{
    // accept one parameters
    PyObject * inventoryCapsule;
    // extract it
    if (!PyArg_ParseTuple(args, "O!", &PyCapsule_Type, &inventoryCapsule)) {
        return 0;
    }
    // decapsulate the inventory
    enabled_t * inventory =
        static_cast<enabled_t *>
        (PyCapsule_GetPointer(inventoryCapsule, enabledInventoryCapsuleName));
    // adjust the state
    if (inventory->state()) {
        Py_RETURN_TRUE;
    }

    Py_RETURN_FALSE;
}


// setDisabledState
const char * const
pyre::extensions::journal::
setDisabledState__name__ = "setDisabledState";

const char * const
pyre::extensions::journal::
setDisabledState__doc__ = "set the state of a normally disabled channel";

PyObject *
pyre::extensions::journal::
setDisabledState(PyObject *, PyObject * args)
{
    // accept two parameters
    PyObject * state;
    PyObject * inventoryCapsule;
    // extract them
    if (!PyArg_ParseTuple(
                          args, "O!O!",
                          &PyCapsule_Type, &inventoryCapsule,
                          &PyBool_Type, &state)) {
        return 0;
    }
    // decapsulate the inventory
    disabled_t * inventory =
        static_cast<disabled_t *>
        (PyCapsule_GetPointer(inventoryCapsule, disabledInventoryCapsuleName));
    // adjust the state
    if (state == Py_True) {
        inventory->activate();
    } else {
        inventory->deactivate();
    }

    Py_INCREF(Py_None);
    return Py_None;
}


// getDisabledState
const char * const
pyre::extensions::journal::
getDisabledState__name__ = "getDisabledState";

const char * const
pyre::extensions::journal::
getDisabledState__doc__ = "get the state of a normally disabled channel";

PyObject *
pyre::extensions::journal::
getDisabledState(PyObject *, PyObject * args)
{
    // accept one parameters
    PyObject * inventoryCapsule;
    // extract it
    if (!PyArg_ParseTuple(args, "O!", &PyCapsule_Type, &inventoryCapsule)) {
        return 0;
    }
    // decapsulate the inventory
    disabled_t * inventory =
        static_cast<disabled_t *>
        (PyCapsule_GetPointer(inventoryCapsule, disabledInventoryCapsuleName));
    // adjust the state
    if (inventory->state()) {
        Py_RETURN_TRUE;
    }

    Py_RETURN_FALSE;
}


// end of file
