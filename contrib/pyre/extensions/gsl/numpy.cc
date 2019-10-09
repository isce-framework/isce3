// -*- C++ -*-
//
// Lijun Zhu (ljzhu@gps.caltech.edu)
//
// (c) 1998-2019 all rights reserved
//


#include <portinfo>
#include <Python.h>
#include <sstream>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "numpy.h"
#include "capsules.h"


const char * const gsl::vector::ndarray__name__ = "vector_ndarray";
const char * const gsl::vector::ndarray__doc__ = "return a numpy array reference of vector";

PyObject *
gsl::vector::ndarray(PyObject *, PyObject * args) {
    // the arguments
    PyObject * self;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!:vector_dataptr",
                                  &PyCapsule_Type, &self);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(self, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    gsl_vector * v = static_cast<gsl_vector *>(PyCapsule_GetPointer(self, gsl::vector::capsule_t));

    // call numpy c api to create a ndarray reference
    import_array(); // must be called for using numpy c-api
    int nd = 1; // ndim
    npy_intp dims[1] = {(npy_intp)v->size}; // shape
    int typenum = NPY_DOUBLE;  // dtype
    PyObject* ndarray = PyArray_SimpleNewFromData(nd, dims, typenum, (void *)v->data);
    // return the ndarray
    return ndarray;
}

const char * const gsl::matrix::ndarray__name__ = "matrix_ndarray";
const char * const gsl::matrix::ndarray__doc__ = "return a numpy array reference of matrix";

PyObject *
gsl::matrix::ndarray(PyObject *, PyObject * args) {
    // the arguments
    PyObject * self;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!:vector_dataptr",
                                  &PyCapsule_Type, &self);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(self, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule");
        return 0;
    }

    // get the matrix
    gsl_matrix * m = static_cast<gsl_matrix *>(PyCapsule_GetPointer(self, gsl::matrix::capsule_t));

    // check whether memory is contiguous
    if(m->tda != m->size2) {
        PyErr_SetString(PyExc_TypeError, "non-contiguous matrix not supported");
        return 0;
    }

    // call numpy c api to create a ndarray reference
    import_array(); // must be called for using numpy c-api
    int nd = 2; // ndim
    npy_intp dims[2] = {(npy_intp)m->size1, (npy_intp)m->size2}; // shape
    int typenum = NPY_DOUBLE;  // dtype
    PyObject* ndarray = PyArray_SimpleNewFromData(nd, dims, typenum, (void *)m->data);
    // return the ndarray
    return ndarray;
}


// end of file
