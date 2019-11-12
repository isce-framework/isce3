// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//


#include <portinfo>
#include <Python.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

// local includes
#include "linalg.h"
#include "capsules.h"


// LU_decomp
const char * const gsl::linalg::LU_decomp__name__ = "linalg_LU_decomp";
const char * const gsl::linalg::LU_decomp__doc__ = "compute the LU decomposition of a matrix";

PyObject *
gsl::linalg::LU_decomp(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:linalg_LU_decomp", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the argument must be a matrix");
        return 0;
    }
    // get the matrix
    gsl_matrix * m =
        static_cast<gsl_matrix *>(PyCapsule_GetPointer(capsule, gsl::matrix::capsule_t));

    // the other arguments
    int sign;
    // allocate a permutation
    gsl_permutation * p = gsl_permutation_alloc(m->size1);
    // compute the decomposition
    gsl_linalg_LU_decomp(m, p, &sign);

    // adjust the reference count of the matrix capsule
    Py_INCREF(capsule);
    // return a tuple
    PyObject * answer = PyTuple_New(3);
    PyTuple_SET_ITEM(answer, 0, capsule);
    PyTuple_SET_ITEM(answer, 1,
                     PyCapsule_New(p, gsl::permutation::capsule_t, gsl::permutation::free));
    PyTuple_SET_ITEM(answer, 2, PyLong_FromLong(sign));

    // and return
    return answer;
}


// LU_invert
const char * const gsl::linalg::LU_invert__name__ = "linalg_LU_invert";
const char * const gsl::linalg::LU_invert__doc__ = "invert a matrix from its LU decomposition";

PyObject *
gsl::linalg::LU_invert(PyObject *, PyObject * args) {
    // the arguments
    PyObject * mcapsule;
    PyObject * pcapsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:linalg_LU_invert",
                                  &PyCapsule_Type, &mcapsule,
                                  &PyCapsule_Type, &pcapsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(mcapsule, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a matrix");
        return 0;
    }
    if (!PyCapsule_IsValid(pcapsule, gsl::permutation::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the second argument must be a matrix");
        return 0;
    }
    // get the matrix
    gsl_matrix * m =
        static_cast<gsl_matrix *>(PyCapsule_GetPointer(mcapsule, gsl::matrix::capsule_t));
    // get the permutation
    gsl_permutation * p =
        static_cast<gsl_permutation *>(PyCapsule_GetPointer(pcapsule, gsl::permutation::capsule_t));
    // allocate space for the inverse
    gsl_matrix * inverse = gsl_matrix_alloc(m->size1, m->size2);

    // compute the inverse
    gsl_linalg_LU_invert(m, p, inverse);

    // and return
    return PyCapsule_New(inverse, gsl::matrix::capsule_t, gsl::matrix::free);
}


// LU_det
const char * const gsl::linalg::LU_det__name__ = "linalg_LU_det";
const char * const gsl::linalg::LU_det__doc__ =
    "compute the determinant of a matrix from its LU decomposition";

PyObject *
gsl::linalg::LU_det(PyObject *, PyObject * args) {
    // the arguments
    int sign;
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!i:linalg_LU_det", &PyCapsule_Type, &capsule, &sign);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a matrix");
        return 0;
    }
    // get the matrix
    gsl_matrix * m =
        static_cast<gsl_matrix *>(PyCapsule_GetPointer(capsule, gsl::matrix::capsule_t));

    // compute the determinant
    double det =  gsl_linalg_LU_det(m, sign);

    // compute the determinant and return
    return PyFloat_FromDouble(det);
}


// LU_lndet
const char * const gsl::linalg::LU_lndet__name__ = "linalg_LU_lndet";
const char * const gsl::linalg::LU_lndet__doc__ =
    "compute the determinant of a matrix from its LU decomposition";

PyObject *
gsl::linalg::LU_lndet(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:linalg_LU_lndet", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a matrix");
        return 0;
    }
    // get the matrix
    gsl_matrix * m =
        static_cast<gsl_matrix *>(PyCapsule_GetPointer(capsule, gsl::matrix::capsule_t));

    // compute the log of the determinant
    double lndet = gsl_linalg_LU_lndet(m);

    // compute the determinant and return
    return PyFloat_FromDouble(lndet);
}


// linalg::cholesky_decomp
const char * const gsl::linalg::cholesky_decomp__name__ = "linalg_cholesky_decomp";
const char * const gsl::linalg::cholesky_decomp__doc__ =
    "compute the Cholesky decomposition of a matrix";

PyObject *
gsl::linalg::cholesky_decomp(PyObject *, PyObject * args) {
    // the arguments
    PyObject * capsule;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:linalg_cholesky_decomp", &PyCapsule_Type, &capsule);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(capsule, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the argument must be a matrix");
        return 0;
    }
    // get the matrix
    gsl_matrix * m =
        static_cast<gsl_matrix *>(PyCapsule_GetPointer(capsule, gsl::matrix::capsule_t));

    // compute the decomposition
    gsl_linalg_cholesky_decomp(m);

    // and return
    Py_INCREF(Py_None);
    return Py_None;
}


// end of file
