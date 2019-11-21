// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//


#include <portinfo>
#include <Python.h>
#include <gsl/gsl_permutation.h>

#include "permutation.h"
#include "capsules.h"


// construction
const char * const gsl::permutation::alloc__name__ = "permutation_alloc";
const char * const gsl::permutation::alloc__doc__ = "allocate a permutation";

PyObject *
gsl::permutation::alloc(PyObject *, PyObject * args) {
    // place holders for the python arguments
    size_t shape;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "k:permutation_alloc", &shape);
    // if something went wrong
    if (!status) return 0;

    // allocate a permutation
    gsl_permutation * v = gsl_permutation_alloc(shape);

    // wrap it in a capsule and return it
    return PyCapsule_New(v, capsule_t, free);
}


// initialization
const char * const gsl::permutation::init__name__ = "permutation_init";
const char * const gsl::permutation::init__doc__ = "initialize a permutation";

PyObject *
gsl::permutation::init(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:permutation_init", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid permutation capsule");
        return 0;
    }

    // get the permutation
    gsl_permutation * p = static_cast<gsl_permutation *>(PyCapsule_GetPointer(capsule, capsule_t));
    // initialize it
    gsl_permutation_init(p);

    // std::cout << "permutation: ";
    // for (size_t i=0; i<gsl_permutation_size(p); i++) {
        // std::cout << " " << gsl_permutation_get(p, i);
    // }
    // std::cout << std::endl;

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


// copy
const char * const gsl::permutation::copy__name__ = "permutation_copy";
const char * const gsl::permutation::copy__doc__ = "build a copy of a permutation";

PyObject *
gsl::permutation::copy(PyObject *, PyObject * args) {
    // the arguments
    PyObject * sourceCapsule;
    PyObject * destinationCapsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:permutation_copy",
                                  &PyCapsule_Type, &destinationCapsule,
                                  &PyCapsule_Type, &sourceCapsule
                                  );
    // if something went wrong
    if (!status) return 0;
    // bail out if the source capsule is not valid
    if (!PyCapsule_IsValid(sourceCapsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid permutation capsule for source");
        return 0;
    }
    // bail out if the destination capsule is not valid
    if (!PyCapsule_IsValid(destinationCapsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid permutation capsule for destination");
        return 0;
    }

    // get the permutations
    gsl_permutation * source =
        static_cast<gsl_permutation *>(PyCapsule_GetPointer(sourceCapsule, capsule_t));
    gsl_permutation * destination =
        static_cast<gsl_permutation *>(PyCapsule_GetPointer(destinationCapsule, capsule_t));
    // copy the data
    gsl_permutation_memcpy(destination, source);

    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


// get
const char * const gsl::permutation::get__name__ = "permutation_get";
const char * const gsl::permutation::get__doc__ = "get the value of a permutation element";

PyObject *
gsl::permutation::get(PyObject *, PyObject * args) {
    // the arguments
    size_t index;
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!k:permutation_get", &PyCapsule_Type, &capsule, &index);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid permutation capsule");
        return 0;
    }

    // get the permutation
    gsl_permutation * p = static_cast<gsl_permutation *>(PyCapsule_GetPointer(capsule, capsule_t));

    // get the value
    size_t value = gsl_permutation_get(p, index);

    // return the value
    return PyLong_FromSize_t(value);
}


// swap
const char * const gsl::permutation::swap__name__ = "permutation_swap";
const char * const gsl::permutation::swap__doc__ = "swap the value of a permutation element";

PyObject *
gsl::permutation::swap(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    size_t index1, index2;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!kk:permutation_swap",
                                  &PyCapsule_Type, &capsule,
                                  &index1, &index2);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid permutation capsule");
        return 0;
    }

    // swap the permutation
    gsl_permutation * p = static_cast<gsl_permutation *>(PyCapsule_GetPointer(capsule, capsule_t));
    // swap the value
    gsl_permutation_swap(p, index1, index2);

    // return the value
    Py_INCREF(Py_None);
    return Py_None;
}


// size
const char * const gsl::permutation::size__name__ = "permutation_size";
const char * const gsl::permutation::size__doc__ = "return the size of a permutation";

PyObject *
gsl::permutation::size(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:permutation_size", &PyCapsule_Type, &capsule);
    // bail out if something went wrong with the argument unpacking
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid permutation capsule");
        return 0;
    }

    // get the permutation
    gsl_permutation * p = static_cast<gsl_permutation *>(PyCapsule_GetPointer(capsule, capsule_t));

    // get the size
    size_t size = gsl_permutation_size(p);

    // and return it
    return PyLong_FromSize_t(size);
}


// valid
const char * const gsl::permutation::valid__name__ = "permutation_valid";
const char * const gsl::permutation::valid__doc__ = "check whether the permutation is valid";

PyObject *
gsl::permutation::valid(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:permutation_valid", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid permutation capsule");
        return 0;
    }

    // get the permutation
    gsl_permutation * p = static_cast<gsl_permutation *>(PyCapsule_GetPointer(capsule, capsule_t));

    // the answer
    PyObject * result = (GSL_SUCCESS == gsl_permutation_valid(p)) ? Py_True : Py_False;

    // return the answer
    Py_INCREF(result);
    return result;
}


// reverse
const char * const gsl::permutation::reverse__name__ = "permutation_reverse";
const char * const gsl::permutation::reverse__doc__ = "reverse a permutation";

PyObject *
gsl::permutation::reverse(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:permutation_reverse", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid permutation capsule");
        return 0;
    }

    // get the permutation
    gsl_permutation * p = static_cast<gsl_permutation *>(PyCapsule_GetPointer(capsule, capsule_t));
    // reverse it
    gsl_permutation_reverse(p);

    // return
    Py_INCREF(Py_None);
    return Py_None;
}


// inverse
const char * const gsl::permutation::inverse__name__ = "permutation_inverse";
const char * const gsl::permutation::inverse__doc__ = "invert a permutation";

PyObject *
gsl::permutation::inverse(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:permutation_inverse", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid permutation capsule");
        return 0;
    }

    // get the permutation
    gsl_permutation * p = static_cast<gsl_permutation *>(PyCapsule_GetPointer(capsule, capsule_t));
    // build the result
    gsl_permutation * inv = gsl_permutation_alloc(gsl_permutation_size(p));
    // invert
    gsl_permutation_inverse(inv, p);

    // wrap it in a capsule and return it
    return PyCapsule_New(inv, capsule_t, free);
}


// next
const char * const gsl::permutation::next__name__ = "permutation_next";
const char * const gsl::permutation::next__doc__ = "compute the next permutation";

PyObject *
gsl::permutation::next(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:permutation_next", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid permutation capsule");
        return 0;
    }

    // get the permutation
    gsl_permutation * p = static_cast<gsl_permutation *>(PyCapsule_GetPointer(capsule, capsule_t));

    // compute the next one
    PyObject * result = (GSL_SUCCESS == gsl_permutation_next(p)) ? Py_True : Py_False;

    // return the value
    Py_INCREF(result);
    return result;
}


// prev
const char * const gsl::permutation::prev__name__ = "permutation_prev";
const char * const gsl::permutation::prev__doc__ = "compute the prev permutation";

PyObject *
gsl::permutation::prev(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:permutation_prev", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid permutation capsule");
        return 0;
    }

    // get the permutation
    gsl_permutation * p = static_cast<gsl_permutation *>(PyCapsule_GetPointer(capsule, capsule_t));

    // compute the prev one
    PyObject * result = (GSL_SUCCESS == gsl_permutation_prev(p)) ? Py_True : Py_False;

    // return the value
    Py_INCREF(result);
    return result;
}


// destructor
void
gsl::permutation::free(PyObject * capsule)
{
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, gsl::permutation::capsule_t)) return;
    // get the permutation
    gsl_permutation * v =
        static_cast<gsl_permutation *>(PyCapsule_GetPointer(capsule, gsl::permutation::capsule_t));
    // deallocate
    gsl_permutation_free(v);
    // and return
    return;
}


// end of file
