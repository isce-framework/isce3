// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//


#include <portinfo>
#include <Python.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

// local includes
#include "blas.h"
#include "capsules.h"


// level 1
// blas::ddot
const char * const gsl::blas::ddot__name__ = "blas_ddot";
const char * const gsl::blas::ddot__doc__ = "compute the scalar product of two vectors";

PyObject *
gsl::blas::ddot(PyObject *, PyObject * args) {
    // the arguments
    PyObject * v1c;
    PyObject * v2c;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:blas_ddot",
                                  &PyCapsule_Type, &v1c, &PyCapsule_Type, &v2c);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(v1c, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a vector");
        return 0;
    }
    if (!PyCapsule_IsValid(v2c, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the second argument must be a vector");
        return 0;
    }

    // get the two vectors
    gsl_vector * v1 = static_cast<gsl_vector *>(PyCapsule_GetPointer(v1c, gsl::vector::capsule_t));
    gsl_vector * v2 = static_cast<gsl_vector *>(PyCapsule_GetPointer(v2c, gsl::vector::capsule_t));
    // the result
    double result;
    // compute the dot product
    gsl_blas_ddot(v1, v2, &result);
    // and return the result
    return PyFloat_FromDouble(result);
}


// blas::dnrm2
const char * const gsl::blas::dnrm2__name__ = "blas_dnrm2";
const char * const gsl::blas::dnrm2__doc__ = "compute the Euclidean norm of a vector";

PyObject *
gsl::blas::dnrm2(PyObject *, PyObject * args) {
    // the arguments
    PyObject * vc;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:blas_dnrm2", &PyCapsule_Type, &vc);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(vc, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    gsl_vector * v = static_cast<gsl_vector *>(PyCapsule_GetPointer(vc, gsl::vector::capsule_t));

    double norm;
    // compute the norm
    norm = gsl_blas_dnrm2(v);

    // and return the result
    return PyFloat_FromDouble(norm);
}


// blas::dasum
const char * const gsl::blas::dasum__name__ = "blas_dasum";
const char * const gsl::blas::dasum__doc__ =
    "compute the sum of the absolute values of the vector entries";

PyObject *
gsl::blas::dasum(PyObject *, PyObject * args) {
    // the arguments
    PyObject * vc;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:blas_dasum", &PyCapsule_Type, &vc);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(vc, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    gsl_vector * v = static_cast<gsl_vector *>(PyCapsule_GetPointer(vc, gsl::vector::capsule_t));

    double norm;

    // compute the norm
    norm = gsl_blas_dasum(v);

    // and return the result
    return PyFloat_FromDouble(norm);
}


// blas::idamax
const char * const gsl::blas::idamax__name__ = "blas_idamax";
const char * const gsl::blas::idamax__doc__ =
    "find the index of the largest element in a vector";

PyObject *
gsl::blas::idamax(PyObject *, PyObject * args) {
    // the arguments
    PyObject * vc;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "O!:blas_idamax", &PyCapsule_Type, &vc);
    // if something went wrong
    if (!status) return 0;
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(vc, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule");
        return 0;
    }

    // get the vector
    gsl_vector * v = static_cast<gsl_vector *>(PyCapsule_GetPointer(vc, gsl::vector::capsule_t));

    CBLAS_INDEX_t index;
    // compute the index
    index = gsl_blas_idamax(v);

    // and return the result
    return PyLong_FromLong(index);
}


// blas::dswap
const char * const gsl::blas::dswap__name__ = "blas_dswap";
const char * const gsl::blas::dswap__doc__ = "swap the contents of two vectors";

PyObject *
gsl::blas::dswap(PyObject *, PyObject * args) {
    // the arguments
    PyObject * xc;
    PyObject * yc;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args,
                                  "O!:blas_dswap",
                                  &PyCapsule_Type, &xc,
                                  &PyCapsule_Type, &yc
                                  );
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(xc, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a vector");
        return 0;
    }
    if (!PyCapsule_IsValid(yc, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the second argument must be a vector");
        return 0;
    }

    // get the vectors
    gsl_vector * x = static_cast<gsl_vector *>(PyCapsule_GetPointer(xc, gsl::vector::capsule_t));
    gsl_vector * y = static_cast<gsl_vector *>(PyCapsule_GetPointer(yc, gsl::vector::capsule_t));

    // swap the contents
    gsl_blas_dswap(x, y);

    // and return
    Py_INCREF(Py_None);
    return Py_None;
}


// blas::dcopy
const char * const gsl::blas::dcopy__name__ = "blas_dcopy";
const char * const gsl::blas::dcopy__doc__ = "copy the contents of one vector into another";

PyObject *
gsl::blas::dcopy(PyObject *, PyObject * args) {
    // the arguments
    PyObject * xc;
    PyObject * yc;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args,
                                  "O!:blas_dcopy",
                                  &PyCapsule_Type, &xc,
                                  &PyCapsule_Type, &yc
                                  );
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(xc, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a vector");
        return 0;
    }
    if (!PyCapsule_IsValid(yc, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the second argument must be a vector");
        return 0;
    }

    // get the vectors
    gsl_vector * x = static_cast<gsl_vector *>(PyCapsule_GetPointer(xc, gsl::vector::capsule_t));
    gsl_vector * y = static_cast<gsl_vector *>(PyCapsule_GetPointer(yc, gsl::vector::capsule_t));

    // copy
    gsl_blas_dcopy(x, y);

    // and return
    Py_INCREF(Py_None);
    return Py_None;
}


// blas::daxpy
const char * const gsl::blas::daxpy__name__ = "blas_daxpy";
const char * const gsl::blas::daxpy__doc__ = "compute the scalar product of two vectors";

PyObject *
gsl::blas::daxpy(PyObject *, PyObject * args) {
    // the arguments
    double a;
    PyObject * v1c;
    PyObject * v2c;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "dO!O!:blas_daxpy",
                                  &a, &PyCapsule_Type, &v1c, &PyCapsule_Type, &v2c);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(v1c, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the second argument must be a vector");
        return 0;
    }
    if (!PyCapsule_IsValid(v2c, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the third argument must be a vector");
        return 0;
    }

    // get the two vectors
    gsl_vector * v1 = static_cast<gsl_vector *>(PyCapsule_GetPointer(v1c, gsl::vector::capsule_t));
    gsl_vector * v2 = static_cast<gsl_vector *>(PyCapsule_GetPointer(v2c, gsl::vector::capsule_t));

    // compute the form
    gsl_blas_daxpy(a, v1, v2);

    // and return
    Py_INCREF(Py_None);
    return Py_None;
}


// blas::dscal
const char * const gsl::blas::dscal__name__ = "blas_dscal";
const char * const gsl::blas::dscal__doc__ = "scale a vector by a number";

PyObject *
gsl::blas::dscal(PyObject *, PyObject * args) {
    // the arguments
    double a;
    PyObject * vc;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "dO!:blas_dscal",
                                  &a,
                                  &PyCapsule_Type, &vc);
    // if something went wrong
    if (!status) return 0;
    // check that the capsule is valid
    if (!PyCapsule_IsValid(vc, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the second argument must be a vector");
        return 0;
    }

    // get the two vectors
    gsl_vector * v = static_cast<gsl_vector *>(PyCapsule_GetPointer(vc, gsl::vector::capsule_t));

    // compute the form
    gsl_blas_dscal(a, v);

    // and return
    Py_INCREF(Py_None);
    return Py_None;
}


// blas::drotg
const char * const gsl::blas::drotg__name__ = "blas_drotg";
const char * const gsl::blas::drotg__doc__ = "compute the Givens rotation for two vectors";

PyObject *
gsl::blas::drotg(PyObject *, PyObject * args) {
    // the arguments
    double x, y;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(args, "dd:blas_drotg", &x, &y);

    // if something went wrong
    if (!status) return 0;

    double c, s;
    // compute the rotation
    gsl_blas_drotg(&x, &y, &c, &s);

    // build a tuple to hold the results
    PyObject * result = PyTuple_New(4);
    PyTuple_SET_ITEM(result, 0, PyFloat_FromDouble(x));
    PyTuple_SET_ITEM(result, 1, PyFloat_FromDouble(y));
    PyTuple_SET_ITEM(result, 2, PyFloat_FromDouble(c));
    PyTuple_SET_ITEM(result, 3, PyFloat_FromDouble(c));

    // and return
    return result;
}


// blas::drot
const char * const gsl::blas::drot__name__ = "blas_drot";
const char * const gsl::blas::drot__doc__ = "apply a Givens rotation to two vectors";

PyObject *
gsl::blas::drot(PyObject *, PyObject * args) {
    // the arguments
    double c, s;
    PyObject * v1c;
    PyObject * v2c;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!dd:blas_drot",
                                  &PyCapsule_Type, &v1c,
                                  &PyCapsule_Type, &v2c,
                                  &c, &s);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(v1c, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a vector");
        return 0;
    }
    if (!PyCapsule_IsValid(v2c, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the second argument must be a vector");
        return 0;
    }

    // get the two vectors
    gsl_vector * v1 = static_cast<gsl_vector *>(PyCapsule_GetPointer(v1c, gsl::vector::capsule_t));
    gsl_vector * v2 = static_cast<gsl_vector *>(PyCapsule_GetPointer(v2c, gsl::vector::capsule_t));

    // compute the form
    gsl_blas_drot(v1, v2, c, s);

    // and return
    Py_INCREF(Py_None);
    return Py_None;
}


// level 2
// blas::dgemv
const char * const gsl::blas::dgemv__name__ = "blas_dgemv";
const char * const gsl::blas::dgemv__doc__ = "compute y = a op(A) x + b y";

PyObject *
gsl::blas::dgemv(PyObject *, PyObject * args) {
    // the arguments
    int op;
    double a, b;
    PyObject * xc;
    PyObject * yc;
    PyObject * Ac;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "idO!O!dO!:blas_dgemv",
                                  &op,
                                  &a,
                                  &PyCapsule_Type, &Ac,
                                  &PyCapsule_Type, &xc,
                                  &b,
                                  &PyCapsule_Type, &yc);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(Ac, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the third argument must be a matrix");
        return 0;
    }
    if (!PyCapsule_IsValid(xc, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the fourth argument must be a vector");
        return 0;
    }
    if (!PyCapsule_IsValid(yc, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the sixth argument must be a vector");
        return 0;
    }

    // decode the enum
    CBLAS_TRANSPOSE_t ctran;
    switch(op) {
    case 0:
        ctran = CblasNoTrans; break;
    case 1:
        ctran = CblasTrans; break;
    case 2:
        ctran = CblasConjTrans; break;
    default:
        PyErr_SetString(PyExc_TypeError, "bad operation flag");
        return 0;
    }
    // get the two vectors
    gsl_vector * x = static_cast<gsl_vector *>(PyCapsule_GetPointer(xc, gsl::vector::capsule_t));
    gsl_vector * y = static_cast<gsl_vector *>(PyCapsule_GetPointer(yc, gsl::vector::capsule_t));
    // get the matrix
    gsl_matrix * A = static_cast<gsl_matrix *>(PyCapsule_GetPointer(Ac, gsl::matrix::capsule_t));

    // compute the form
    gsl_blas_dgemv(ctran, a, A, x, b, y);

    // and return
    Py_INCREF(Py_None);
    return Py_None;
}


// blas::dtrmv
const char * const gsl::blas::dtrmv__name__ = "blas_dtrmv";
const char * const gsl::blas::dtrmv__doc__ = "compute x = op(A) x";

PyObject *
gsl::blas::dtrmv(PyObject *, PyObject * args) {
    // the arguments
    int uplo, op, unitDiag;
    PyObject * xc;
    PyObject * Ac;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "iiiO!O!:blas_dtrmv",
                                  &uplo, &op, &unitDiag,
                                  &PyCapsule_Type, &Ac,
                                  &PyCapsule_Type, &xc);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(Ac, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the fourth argument must be a matrix");
        return 0;
    }
    if (!PyCapsule_IsValid(xc, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the fifth argument must be a vector");
        return 0;
    }

    // decode the enums
    CBLAS_UPLO_t cuplo = uplo ? CblasUpper : CblasLower;
    CBLAS_DIAG_t cdiag = unitDiag ? CblasUnit : CblasNonUnit;
    CBLAS_TRANSPOSE_t ctran;
    switch(op) {
    case 0:
        ctran = CblasNoTrans; break;
    case 1:
        ctran = CblasTrans; break;
    case 2:
        ctran = CblasConjTrans; break;
    default:
        PyErr_SetString(PyExc_TypeError, "bad operation flag");
        return 0;
    }

    // get the two vectors
    gsl_vector * x = static_cast<gsl_vector *>(PyCapsule_GetPointer(xc, gsl::vector::capsule_t));
    // get the matrix
    gsl_matrix * A = static_cast<gsl_matrix *>(PyCapsule_GetPointer(Ac, gsl::matrix::capsule_t));

    // compute the form
    gsl_blas_dtrmv(cuplo, ctran, cdiag, A, x);

    // and return
    Py_INCREF(Py_None);
    return Py_None;
}


// blas::dtrsv
const char * const gsl::blas::dtrsv__name__ = "blas_dtrsv";
const char * const gsl::blas::dtrsv__doc__ = "compute x = inv(op(A)) x";

PyObject *
gsl::blas::dtrsv(PyObject *, PyObject * args) {
    // the arguments
    int uplo, op, unitDiag;
    PyObject * xc;
    PyObject * Ac;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "iiiO!O!:blas_dtrsv",
                                  &uplo, &op, &unitDiag,
                                  &PyCapsule_Type, &Ac,
                                  &PyCapsule_Type, &xc);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(Ac, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the fourth argument must be a matrix");
        return 0;
    }
    if (!PyCapsule_IsValid(xc, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the fifth argument must be a vector");
        return 0;
    }

    // decode the enums
    CBLAS_UPLO_t cuplo = uplo ? CblasUpper : CblasLower;
    CBLAS_DIAG_t cdiag = unitDiag ? CblasUnit : CblasNonUnit;
    CBLAS_TRANSPOSE_t ctran;
    switch(op) {
    case 0:
        ctran = CblasNoTrans; break;
    case 1:
        ctran = CblasTrans; break;
    case 2:
        ctran = CblasConjTrans; break;
    default:
        PyErr_SetString(PyExc_TypeError, "bad operation flag");
        return 0;
    }

    // get the two vectors
    gsl_vector * x = static_cast<gsl_vector *>(PyCapsule_GetPointer(xc, gsl::vector::capsule_t));
    // get the matrix
    gsl_matrix * A = static_cast<gsl_matrix *>(PyCapsule_GetPointer(Ac, gsl::matrix::capsule_t));

    // compute the form
    gsl_blas_dtrsv(cuplo, ctran, cdiag, A, x);

    // and return
    Py_INCREF(Py_None);
    return Py_None;
}


// blas::dsymv
const char * const gsl::blas::dsymv__name__ = "blas_dsymv";
const char * const gsl::blas::dsymv__doc__ = "compute y = a A x + b y";

PyObject *
gsl::blas::dsymv(PyObject *, PyObject * args) {
    // the arguments
    int uplo;
    double a, b;
    PyObject * xc;
    PyObject * yc;
    PyObject * Ac;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "idO!O!dO!:blas_dsymv",
                                  &uplo,
                                  &a,
                                  &PyCapsule_Type, &Ac,
                                  &PyCapsule_Type, &xc,
                                  &b,
                                  &PyCapsule_Type, &yc);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(Ac, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the third argument must be a matrix");
        return 0;
    }
    if (!PyCapsule_IsValid(xc, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the fourth argument must be a vector");
        return 0;
    }
    if (!PyCapsule_IsValid(yc, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the sixth argument must be a vector");
        return 0;
    }

    // decode the enum
    CBLAS_UPLO_t cuplo = uplo ? CblasUpper : CblasLower;
    // get the two vectors
    gsl_vector * x = static_cast<gsl_vector *>(PyCapsule_GetPointer(xc, gsl::vector::capsule_t));
    gsl_vector * y = static_cast<gsl_vector *>(PyCapsule_GetPointer(yc, gsl::vector::capsule_t));
    // get the matrix
    gsl_matrix * A = static_cast<gsl_matrix *>(PyCapsule_GetPointer(Ac, gsl::matrix::capsule_t));

    // compute the form
    gsl_blas_dsymv(cuplo, a, A, x, b, y);

    // and return
    Py_INCREF(Py_None);
    return Py_None;
}


// blas::dsyr
const char * const gsl::blas::dsyr__name__ = "blas_dsyr";
const char * const gsl::blas::dsyr__doc__ = "compute A = a x x^T + A";

PyObject *
gsl::blas::dsyr(PyObject *, PyObject * args) {
    // the arguments
    int uplo;
    double a;
    PyObject * xc;
    PyObject * Ac;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "idO!O!:blas_dsyr",
                                  &uplo,
                                  &a,
                                  &PyCapsule_Type, &xc,
                                  &PyCapsule_Type, &Ac);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(Ac, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the fourth argument must be a matrix");
        return 0;
    }
    if (!PyCapsule_IsValid(xc, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the third argument must be a vector");
        return 0;
    }

    // decode the enum
    CBLAS_UPLO_t cuplo = uplo ? CblasUpper : CblasLower;
    // get the two vectors
    gsl_vector * x = static_cast<gsl_vector *>(PyCapsule_GetPointer(xc, gsl::vector::capsule_t));
    // get the matrix
    gsl_matrix * A = static_cast<gsl_matrix *>(PyCapsule_GetPointer(Ac, gsl::matrix::capsule_t));

    // compute the form
    gsl_blas_dsyr(cuplo, a, x, A);

    // and return
    Py_INCREF(Py_None);
    return Py_None;
}


// blas::dgemm
const char * const gsl::blas::dgemm__name__ = "blas_dgemm";
const char * const gsl::blas::dgemm__doc__ = "compute y = a op(A) x + b y";

PyObject *
gsl::blas::dgemm(PyObject *, PyObject * args) {
    // the arguments
    int opA, opB;
    double a, b;
    PyObject * Ac;
    PyObject * Bc;
    PyObject * Cc;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "iidO!O!dO!:blas_dgemm",
                                  &opA, &opB,
                                  &a,
                                  &PyCapsule_Type, &Ac,
                                  &PyCapsule_Type, &Bc,
                                  &b,
                                  &PyCapsule_Type, &Cc);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(Ac, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the fourth argument must be a matrix");
        return 0;
    }
    if (!PyCapsule_IsValid(Bc, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the fifth argument must be a matrix");
        return 0;
    }
    if (!PyCapsule_IsValid(Cc, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the seventh argument must be a matrix");
        return 0;
    }

    // decode the enum
    CBLAS_TRANSPOSE_t ctranA;
    switch(opA) {
    case 0:
        ctranA = CblasNoTrans; break;
    case 1:
        ctranA = CblasTrans; break;
    case 2:
        ctranA = CblasConjTrans; break;
    default:
        PyErr_SetString(PyExc_TypeError, "bad operation flag");
        return 0;
    }
    // decode the other enum
    CBLAS_TRANSPOSE_t ctranB;
    switch(opB) {
    case 0:
        ctranB = CblasNoTrans; break;
    case 1:
        ctranB = CblasTrans; break;
    case 2:
        ctranB = CblasConjTrans; break;
    default:
        PyErr_SetString(PyExc_TypeError, "bad operation flag");
        return 0;
    }

    // get the matrices
    gsl_matrix * A = static_cast<gsl_matrix *>(PyCapsule_GetPointer(Ac, gsl::matrix::capsule_t));
    gsl_matrix * B = static_cast<gsl_matrix *>(PyCapsule_GetPointer(Bc, gsl::matrix::capsule_t));
    gsl_matrix * C = static_cast<gsl_matrix *>(PyCapsule_GetPointer(Cc, gsl::matrix::capsule_t));

    // compute the form
    gsl_blas_dgemm(ctranA, ctranB, a, A, B, b, C);

    // and return
    Py_INCREF(Py_None);
    return Py_None;
}


// blas::dsymm
const char * const gsl::blas::dsymm__name__ = "blas_dsymm";
const char * const gsl::blas::dsymm__doc__ = "compute C = a A B  + b C where A is symmetric";

PyObject *
gsl::blas::dsymm(PyObject *, PyObject * args) {
    // the arguments
    int side, UploA;
    double a, b;
    PyObject * Ac;
    PyObject * Bc;
    PyObject * Cc;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "iidO!O!dO!:blas_dsymm",
                                  &side, &UploA,
                                  &a,
                                  &PyCapsule_Type, &Ac,
                                  &PyCapsule_Type, &Bc,
                                  &b,
                                  &PyCapsule_Type, &Cc);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(Ac, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the fourth argument must be a matrix");
        return 0;
    }
    if (!PyCapsule_IsValid(Bc, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the fifth argument must be a matrix");
        return 0;
    }
    if (!PyCapsule_IsValid(Cc, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the seventh argument must be a matrix");
        return 0;
    }

    // decode the enum
    CBLAS_SIDE_t cside = side ? CblasRight : CblasLeft;
    CBLAS_UPLO_t cuplo = UploA ? CblasUpper : CblasLower;


    // get the matrices
    gsl_matrix * A = static_cast<gsl_matrix *>(PyCapsule_GetPointer(Ac, gsl::matrix::capsule_t));
    gsl_matrix * B = static_cast<gsl_matrix *>(PyCapsule_GetPointer(Bc, gsl::matrix::capsule_t));
    gsl_matrix * C = static_cast<gsl_matrix *>(PyCapsule_GetPointer(Cc, gsl::matrix::capsule_t));

    // compute the form
    gsl_blas_dsymm(cside, cuplo, a, A, B, b, C);

    // and return
    Py_INCREF(Py_None);
    return Py_None;
}



// blas::dtrmm
const char * const gsl::blas::dtrmm__name__ = "blas_dtrmm";
const char * const gsl::blas::dtrmm__doc__ = "compute B = a op(A) B";

PyObject *
gsl::blas::dtrmm(PyObject *, PyObject * args) {
    // the arguments
    double a;
    int side, uplo, op, unitDiag;
    PyObject * Ac;
    PyObject * Bc;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "iiiidO!O!:blas_dtrmm",
                                  &side, &uplo, &op, &unitDiag,
                                  &a,
                                  &PyCapsule_Type, &Ac,
                                  &PyCapsule_Type, &Bc);
    // if something went wrong
    if (!status) return 0;
    // bail out if the two capsules are not valid
    if (!PyCapsule_IsValid(Ac, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the sixth argument must be a matrix");
        return 0;
    }
    if (!PyCapsule_IsValid(Bc, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the seventh argument must be a vector");
        return 0;
    }

    // decode the enums
    CBLAS_SIDE_t cside = side ? CblasRight : CblasLeft;
    CBLAS_UPLO_t cuplo = uplo ? CblasUpper : CblasLower;
    CBLAS_DIAG_t cdiag = unitDiag ? CblasUnit : CblasNonUnit;
    CBLAS_TRANSPOSE_t ctran;
    switch(op) {
    case 0:
        ctran = CblasNoTrans; break;
    case 1:
        ctran = CblasTrans; break;
    case 2:
        ctran = CblasConjTrans; break;
    default:
        PyErr_SetString(PyExc_TypeError, "bad operation flag");
        return 0;
    }

    // get the two matrices
    gsl_matrix * A = static_cast<gsl_matrix *>(PyCapsule_GetPointer(Ac, gsl::matrix::capsule_t));
    gsl_matrix * B = static_cast<gsl_matrix *>(PyCapsule_GetPointer(Bc, gsl::matrix::capsule_t));

    // compute the form
    gsl_blas_dtrmm(cside, cuplo, ctran, cdiag, a, A, B);

    // and return
    Py_INCREF(Py_None);
    return Py_None;
}


// end of file
