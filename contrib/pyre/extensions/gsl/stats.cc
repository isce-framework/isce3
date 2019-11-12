// -*- C++ -*-
//
// michael a.g. aïvázis @ orthologue
// Lijun Zhu @ Caltech
// (c) 1998-2019 all rights reserved
//


#include <portinfo>
#include <Python.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_statistics.h>
#include <cstdio>

// local includes
#include "stats.h"
#include "capsules.h"


// stats::correlation
const char * const gsl::stats::correlation__name__ = "stats_correlation";
const char * const gsl::stats::correlation__doc__ = "Pearson correlation coefficient between the datasets";

PyObject *
gsl::stats::correlation(PyObject *, PyObject * args) {
    // the arguments
    PyObject * v1c;
    PyObject * v2c;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:stats_correlation",
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
    result = gsl_stats_correlation(v1->data, 1, v2->data, 1, v1->size);
    // and return the result
    return PyFloat_FromDouble(result);
}

// stats::covariance
const char * const gsl::stats::covariance__name__ = "stats_covariance";
const char * const gsl::stats::covariance__doc__ = "the covariance of two datasets";

PyObject *
gsl::stats::covariance(PyObject *, PyObject * args) {
    // the arguments
    PyObject * v1c;
    PyObject * v2c;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!O!:stats_correlation",
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
    result = gsl_stats_covariance(v1->data, 1, v2->data, 1, v1->size);
    // and return the result
    return PyFloat_FromDouble(result);
}

// stats::mean
const char * const gsl::stats::matrix_mean__name__ = "stats_matrix_mean";
const char * const gsl::stats::matrix_mean__doc__ = "the mean value(s) of a matrix";

PyObject *
gsl::stats::matrix_mean(PyObject *, PyObject * args)
{
    // the arguments
    PyObject * matrixCapsule; // input matrix
    PyObject *meanCapsule; // output mean vectors
    int axis;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!kO:stats_matrix_mean",
                                  &PyCapsule_Type, &matrixCapsule, &axis, &meanCapsule);
    // if something went wrong
    if (!status) return 0;

    // bail out if the capsule is not valid
    // input matrix capsule
    if (!PyCapsule_IsValid(matrixCapsule, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, " invalid matrix capsule");
        return 0;
    }
    // output mean capsules
    if (!PyCapsule_IsValid(meanCapsule, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, " invalid vector capsule for mean");
        return 0;
    }

    // get the matrix/vector from capsules
    gsl_matrix * m = static_cast<gsl_matrix *>(PyCapsule_GetPointer(matrixCapsule, gsl::matrix::capsule_t));
    gsl_vector * meanVec = static_cast<gsl_vector *>(PyCapsule_GetPointer(meanCapsule, gsl::vector::capsule_t));

    // temporary results
    double mean;
    // pointer to data
    double * datap = (double *)m->data;

    size_t rows = m->size1;
    size_t cols = m->size2;
    size_t tda = m->tda;

    // for different axis
    switch(axis) {
        case(0): // along row
            for(size_t i=0; i<cols; ++i)
            {
                // compute mean for each column
                mean =  gsl_stats_mean(datap, tda, rows); // (data[], stride, total elements)
                // set the values to output vectors
                gsl_vector_set(meanVec, i, mean);
                // move pointer to next column
                datap++;
            }
            break;
        case(1): // along column
            for(size_t i=0; i<rows; ++i)
            {
                // compute (mean, sd) for each row
                mean =  gsl_stats_mean(datap, 1, cols);
                // set the val
                gsl_vector_set(meanVec, i, mean);
                // move pointer to next row
                datap += tda;
            }
            break;
        default: // all elements
            if (tda!=cols) {
                PyErr_SetString(PyExc_TypeError, "Not working for non-contiguous matrix!");
                return (PyObject *) NULL;
            }
            mean = gsl_stats_mean(datap, 1, cols*rows);
            gsl_vector_set(meanVec, 0, mean);
    }

    // all done
    // return None
    Py_INCREF(Py_None);
    return Py_None;
}


// stats::matrix_mean_sd
// (sample) standard deviation sd = sqrt{ \sum (x_i- mean)^2 /(N-1)}
const char * const gsl::stats::matrix_mean_sd__name__ = "stats_matrix_mean_sd";
const char * const gsl::stats::matrix_mean_sd__doc__ = "the mean and (sample) standard deviation of a matrix";

PyObject *
gsl::stats::matrix_mean_sd(PyObject *, PyObject * args)
{
    // the arguments
    PyObject * matrixCapsule; // input matrix
    PyObject *meanCapsule, *sdCapsule; // output (mean, sd) vectors
    int axis;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!kOO:stats_matrix_mean_sd",
                                  &PyCapsule_Type, &matrixCapsule, &axis, &meanCapsule, &sdCapsule);
    // if something went wrong
    if (!status) return 0;

    // bail out if the capsule is not valid
    // input matrix capsule
    if (!PyCapsule_IsValid(matrixCapsule, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, " invalid matrix capsule");
        return 0;
    }
    // output mean, sd capsules
    if (!PyCapsule_IsValid(meanCapsule, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, " invalid vector capsule for mean");
        return 0;
    }
    if (!PyCapsule_IsValid(sdCapsule, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule for standard deviation");
        return 0;
    }

    // get the matrix/vector from capsules
    gsl_matrix * m = static_cast<gsl_matrix *>(PyCapsule_GetPointer(matrixCapsule, gsl::matrix::capsule_t));
    gsl_vector * meanVec = static_cast<gsl_vector *>(PyCapsule_GetPointer(meanCapsule, gsl::vector::capsule_t));
    gsl_vector * sdVec = static_cast<gsl_vector *>(PyCapsule_GetPointer(sdCapsule, gsl::vector::capsule_t));

    // temporary results
    double mean, sd;
    // pointer to data
    double * datap = (double *)m->data;

    size_t rows = m->size1;
    size_t cols = m->size2;
    size_t tda = m->tda;

    // for different axis
    switch(axis) {
        case(0): // along row
            for(size_t i=0; i<cols; ++i)
            {
                // compute mean for each column
                mean =  gsl_stats_mean(datap, tda, rows); // (data[], stride, total elements)
                // compute sd for each column
                sd = gsl_stats_sd_m(datap, tda, rows, mean);
                // set the values to output vectors
                gsl_vector_set(meanVec, i, mean);
                gsl_vector_set(sdVec, i, sd);
                // move pointer to next column
                datap++;
            }
            break;
        case(1): // along column
            for(size_t i=0; i<rows; ++i)
            {
                // compute (mean, sd) for each row
                mean =  gsl_stats_mean(datap, 1, cols);
                sd = gsl_stats_sd_m(datap, 1, cols, mean);
                // set the val
                gsl_vector_set(meanVec, i, mean);
                gsl_vector_set(sdVec, i, sd);
                // move pointer to next row
                datap += tda;
            }
            break;
        default: // all elements
            if (tda!=cols) {
                PyErr_SetString(PyExc_TypeError, "Not working for non-contiguous matrix!");
                return (PyObject *) NULL;
            }
            mean = gsl_stats_mean(datap, 1, cols*rows);
            sd = gsl_stats_sd_m(datap, 1, cols*rows, mean);
            gsl_vector_set(meanVec, 0, mean);
            gsl_vector_set(sdVec, 0, sd);
    }

    // all done
    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// stats::matrix_mean_std
// (population) standard deviation sd = sqrt{ \sum (x_i- mean)^2 /N}
const char * const gsl::stats::matrix_mean_std__name__ = "stats_matrix_mean_std";
const char * const gsl::stats::matrix_mean_std__doc__ = "the mean and (population) standard deviation of a matrix";

PyObject *
gsl::stats::matrix_mean_std(PyObject *, PyObject * args)
{
    // the arguments
    PyObject * matrixCapsule; // input matrix
    PyObject *meanCapsule, *sdCapsule; // output (mean, sd) vectors
    int axis;
    // unpack the argument tuple
    int status = PyArg_ParseTuple(
                                  args, "O!kOO:stats_matrix_mean_std",
                                  &PyCapsule_Type, &matrixCapsule, &axis, &meanCapsule, &sdCapsule);
    // if something went wrong
    if (!status) return 0;

    // bail out if the capsule is not valid
    // input matrix capsule
    if (!PyCapsule_IsValid(matrixCapsule, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, " invalid matrix capsule");
        return 0;
    }
    // output mean, sd capsules
    if (!PyCapsule_IsValid(meanCapsule, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, " invalid vector capsule for mean");
        return 0;
    }
    if (!PyCapsule_IsValid(sdCapsule, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule for standard deviation");
        return 0;
    }

    // get the matrix/vector from capsules
    gsl_matrix * m = static_cast<gsl_matrix *>(PyCapsule_GetPointer(matrixCapsule, gsl::matrix::capsule_t));
    gsl_vector * meanVec = static_cast<gsl_vector *>(PyCapsule_GetPointer(meanCapsule, gsl::vector::capsule_t));
    gsl_vector * sdVec = static_cast<gsl_vector *>(PyCapsule_GetPointer(sdCapsule, gsl::vector::capsule_t));

    // temporary results
    double mean, sd;
    // pointer to data
    double * datap = (double *)m->data;

    size_t rows = m->size1;
    size_t cols = m->size2;
    size_t tda = m->tda;

    // for different axis
    switch(axis) {
        case(0): // along row
            for(size_t i=0; i<cols; ++i)
            {
                // compute mean for each column
                mean =  gsl_stats_mean(datap, tda, rows); // (data[], stride, total elements)
                // compute sd for each column
                sd = gsl_stats_sd_with_fixed_mean(datap, tda, rows, mean);
                // set the values to output vectors
                gsl_vector_set(meanVec, i, mean);
                gsl_vector_set(sdVec, i, sd);
                // move pointer to next column
                datap++;
            }
            break;
        case(1): // along column
            for(size_t i=0; i<rows; ++i)
            {
                // compute (mean, sd) for each row
                mean =  gsl_stats_mean(datap, 1, cols);
                sd = gsl_stats_sd_with_fixed_mean(datap, 1, cols, mean);
                // set the val
                gsl_vector_set(meanVec, i, mean);
                gsl_vector_set(sdVec, i, sd);
                // move pointer to next row
                datap += tda;
            }
            break;
        default: // all elements
            if (tda!=cols) {
                PyErr_SetString(PyExc_TypeError, "Not working for non-contiguous matrix!");
                return (PyObject *) NULL;
            }
            mean = gsl_stats_mean(datap, 1, cols*rows);
            sd = gsl_stats_sd_with_fixed_mean(datap, 1, cols*rows, mean);
            gsl_vector_set(meanVec, 0, mean);
            gsl_vector_set(sdVec, 0, sd);
    }

    // all done
    // return None
    Py_INCREF(Py_None);
    return Py_None;
}

// end of file
