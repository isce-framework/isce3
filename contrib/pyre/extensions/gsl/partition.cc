// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#include <portinfo>
#include <Python.h>

// my declarations
#include "partition.h"

// the external libraries
#include <mpi.h>
#include <gsl/gsl_matrix.h>
// the pyre mpi library
#include <pyre/mpi.h>
// the extension info
#include "capsules.h"
#include <pyre/mpi/capsules.h>


// matrix operations
// bcast
const char * const
gsl::mpi::
bcastMatrix__name__ = "bcastMatrix";

const char * const
gsl::mpi::
bcastMatrix__doc__ = "broadcast a matrix to all members of a communicator";

PyObject *
gsl::mpi::
bcastMatrix(PyObject *, PyObject * args)
{
    // place holders
    int source;
    PyObject *communicatorCapsule, *matrixCapsule;

    // parse the argument list
    if (!PyArg_ParseTuple(
                          args,
                          "O!iO:bcastMatrix",
                          &PyCapsule_Type, &communicatorCapsule,
                          &source,
                          &matrixCapsule // don't force the capsule type check; it may be {None}
                          )) {
        return 0;
    }
    // check the communicator capsule
    if (!PyCapsule_IsValid(communicatorCapsule, ::mpi::communicator::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }
    // get the communicator
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>
        (PyCapsule_GetPointer(communicatorCapsule, ::mpi::communicator::capsule_t));

    // the matrix
    gsl_matrix * matrix;
    // the shape of the matrix
    long dim[2];

    // I only have a valid matrix at the {source} rank
    if (comm->rank() == source) {
        // check the matrix capsule
        if (!PyCapsule_IsValid(matrixCapsule, gsl::matrix::capsule_t)) {
            PyErr_SetString(PyExc_TypeError, "invalid matrix capsule for source");
            return 0;
        }
        // get the source matrix
        matrix =
            static_cast<gsl_matrix *>
            (PyCapsule_GetPointer(matrixCapsule, gsl::matrix::capsule_t));
        // fill out the shape of the matrix
        dim[0] = matrix->size1;
        dim[1] = matrix->size2;
    }

    // broadcast the shape
    MPI_Bcast(dim, 2, MPI_LONG, source, comm->handle());

    // unpack the shape
    size_t rows = dim[0];
    size_t columns = dim[1];

    // everybody except the source task
    if (comm->rank() != source) {
        // build the destination matrix
        matrix = gsl_matrix_alloc(rows, columns);
    }

    // extract the pointer to the payload
    double * data = matrix->data;

    // broadcast the data
    int status = MPI_Bcast(data, rows*columns, MPI_DOUBLE, source, comm->handle());

    // check the return code
    if (status != MPI_SUCCESS) {
        // and throw an exception if anything went wrong
        PyErr_SetString(PyExc_RuntimeError, "MPI_Scatter failed");
        return 0;
    }

    // wrap the destination matrix in a capsule and return it
    PyObject * capsule;
    // in the source task
    if (comm->rank() == source) {
        // increment the reference count of the existing capsule
        Py_INCREF(matrixCapsule);
        // and make it the result
        capsule = matrixCapsule;
    } else {
        // everybody else gets a new one
        capsule = PyCapsule_New(matrix, gsl::matrix::capsule_t, gsl::matrix::free);
    }

    // build the matrix shape
    PyObject * shape = PyTuple_New(2);
    PyTuple_SET_ITEM(shape, 0, PyLong_FromLong(rows));
    PyTuple_SET_ITEM(shape, 1, PyLong_FromLong(columns));

    // build the result
    PyObject * result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, capsule);
    PyTuple_SET_ITEM(result, 1, shape);

    // and return it
    return result;
}


// gather
const char * const
gsl::mpi::
gatherMatrix__name__ = "gatherMatrix";

const char * const
gsl::mpi::
gatherMatrix__doc__ = "gather a matrix from the members of a communicator";

PyObject *
gsl::mpi::
gatherMatrix(PyObject *, PyObject * args)
{
    // place holders
    int destination;
    PyObject *communicatorCapsule, *matrixCapsule;

    // parse the argument list
    if (!PyArg_ParseTuple(
                          args,
                          "O!iO!:gatherMatrix",
                          &PyCapsule_Type, &communicatorCapsule,
                          &destination,
                          &PyCapsule_Type, &matrixCapsule)) {
        return 0;
    }
    // check the communicator capsule
    if (!PyCapsule_IsValid(communicatorCapsule, ::mpi::communicator::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }
    // get the communicator
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>
        (PyCapsule_GetPointer(communicatorCapsule, ::mpi::communicator::capsule_t));

    // check the matrix capsule
    if (!PyCapsule_IsValid(matrixCapsule, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid matrix capsule for source");
        return 0;
    }
    // get the source matrix
    gsl_matrix * matrix =
        static_cast<gsl_matrix *>
        (PyCapsule_GetPointer(matrixCapsule, gsl::matrix::capsule_t));

    // the place to deposit the data
    double * data = 0;
    // and the destination matrix
    gsl_matrix * bertha = 0;

    // at the destination task
    if (comm->rank() == destination) {
        // figure out the shape
        int rows = matrix->size1 * comm->size();
        int columns = matrix->size2;
        // build the destination matrix
        bertha = gsl_matrix_alloc(rows, columns);
        // and use its payload as the location to deposit the data
        data = bertha->data;
    }

    // the length of each contribution
    int size = matrix->size1 * matrix->size2;
    // gather the data
    int status = MPI_Gather(
                            matrix->data, size, MPI_DOUBLE, // send buffer
                            data, size, MPI_DOUBLE, // receive buffer
                            destination, comm->handle() // address
                            );

    // check the return code
    if (status != MPI_SUCCESS) {
        // and throw an exception if anything went wrong
        PyErr_SetString(PyExc_RuntimeError, "MPI_Gather failed");
        return 0;
    }

    // at all tasks except the destination task
    if (comm->rank() != destination) {
        // return {None}
        Py_INCREF(Py_None);
        return Py_None;
    }

    // wrap the destination matrix in a capsule
    PyObject * capsule = PyCapsule_New(bertha, gsl::matrix::capsule_t, gsl::matrix::free);

    // build the matrix shape
    PyObject * shape = PyTuple_New(2);
    PyTuple_SET_ITEM(shape, 0, PyLong_FromLong(bertha->size1));
    PyTuple_SET_ITEM(shape, 1, PyLong_FromLong(bertha->size2));

    // build the result
    PyObject * result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, capsule);
    PyTuple_SET_ITEM(result, 1, shape);

    // and return it
    return result;
}


// scatter
const char * const
gsl::mpi::
scatterMatrix__name__ = "scatterMatrix";

const char * const
gsl::mpi::
scatterMatrix__doc__ = "scatter a matrix to the members of a communicator";

PyObject *
gsl::mpi::
scatterMatrix(PyObject *, PyObject * args)
{
    // place holders
    int source;
    PyObject *communicatorCapsule, *matrixCapsule, *destinationCapsule;

    // parse the argument list
    if (!PyArg_ParseTuple(
                          args,
                          "O!iO!O:scatterMatrix",
                          &PyCapsule_Type, &communicatorCapsule,
                          &source,
                          &PyCapsule_Type, &destinationCapsule,
                          &matrixCapsule // don't force the capsule type check; it may be {None}
                          )) {
        return 0;
    }
    // check the communicator capsule
    if (!PyCapsule_IsValid(communicatorCapsule, ::mpi::communicator::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }
    // check the destination capsule
    if (!PyCapsule_IsValid(destinationCapsule, gsl::matrix::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the third argument must be a valid matrix");
        return 0;
    }
    // get the communicator
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>
        (PyCapsule_GetPointer(communicatorCapsule, ::mpi::communicator::capsule_t));
    // get the destination matrix
    gsl_matrix * destination =
        static_cast<gsl_matrix *>
        (PyCapsule_GetPointer(destinationCapsule, gsl::matrix::capsule_t));

    // the pointer to source payload
    double * data = 0;
    // I only have a valid matrix at the {source} rank
    if (comm->rank() == source) {
        // check the matrix capsule
        if (!PyCapsule_IsValid(matrixCapsule, gsl::matrix::capsule_t)) {
            PyErr_SetString(PyExc_TypeError, "invalid matrix capsule for source");
            return 0;
        }
        // get the source matrix
        gsl_matrix * matrix =
            static_cast<gsl_matrix *>
            (PyCapsule_GetPointer(matrixCapsule, gsl::matrix::capsule_t));
        // and extract the pointer to the payload
        data = matrix->data;
    }

    // get the rows and columns of the destination
    int rows = destination->size1;
    int columns = destination->size2;
    // scatter the data
    int status = MPI_Scatter(
                         data, rows*columns, MPI_DOUBLE, // source buffer
                         destination->data, rows*columns, MPI_DOUBLE, // destination buffer
                         source, comm->handle() // address
                         );

    // check the return code
    if (status != MPI_SUCCESS) {
        // and throw an exception if anything went wrong
        PyErr_SetString(PyExc_RuntimeError, "MPI_Scatter failed");
        return 0;
    }

    // all done
    Py_INCREF(Py_None);
    return Py_None;
}


// vector operations
// bcast
const char * const
gsl::mpi::
bcastVector__name__ = "bcastVector";

const char * const
gsl::mpi::
bcastVector__doc__ = "broadcast a vector to all members of a communicator";

PyObject *
gsl::mpi::
bcastVector(PyObject *, PyObject * args)
{
    // place holders
    int source;
    PyObject *communicatorCapsule, *vectorCapsule;

    // parse the argument list
    if (!PyArg_ParseTuple(
                          args,
                          "O!iO:bcastVector",
                          &PyCapsule_Type, &communicatorCapsule,
                          &source,
                          &vectorCapsule // don't force the capsule type check; it may be {None}
                          )) {
        return 0;
    }
    // check the communicator capsule
    if (!PyCapsule_IsValid(communicatorCapsule, ::mpi::communicator::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }
    // get the communicator
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>
        (PyCapsule_GetPointer(communicatorCapsule, ::mpi::communicator::capsule_t));

    // the vector
    gsl_vector * vector;
    // the shape of the vector
    long dim;

    // I only have a valid vector at the {source} rank
    if (comm->rank() == source) {
        // check the vector capsule
        if (!PyCapsule_IsValid(vectorCapsule, gsl::vector::capsule_t)) {
            PyErr_SetString(PyExc_TypeError, "invalid vector capsule for source");
            return 0;
        }
        // get the source vector
        vector =
            static_cast<gsl_vector *>
            (PyCapsule_GetPointer(vectorCapsule, gsl::vector::capsule_t));
        // fill out the shape of the vector
        dim = vector->size;
    }

    // broadcast the shape
    MPI_Bcast(&dim, 1, MPI_LONG, source, comm->handle());

    // everybody except the source task
    if (comm->rank() != source) {
        // build the destination vector
        vector = gsl_vector_alloc(dim);
    }

    // extract the pointer to the payload
    double * data = vector->data;

    // broadcast the data
    int status = MPI_Bcast(data, dim, MPI_DOUBLE, source, comm->handle());

    // check the return code
    if (status != MPI_SUCCESS) {
        // and throw an exception if anything went wrong
        PyErr_SetString(PyExc_RuntimeError, "MPI_Scatter failed");
        return 0;
    }

    // wrap the destination vector in a capsule and return it
    PyObject * capsule;
    // in the source task
    if (comm->rank() == source) {
        // increment the reference count of the existing capsule
        Py_INCREF(vectorCapsule);
        // and make it the result
        capsule = vectorCapsule;
    } else {
        // everybody else gets a new one
        capsule = PyCapsule_New(vector, gsl::vector::capsule_t, gsl::vector::free);
    }

    // build the result
    PyObject * result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, capsule);
    PyTuple_SET_ITEM(result, 1, PyLong_FromLong(dim));

    // and return it
    return result;
}


// gather
const char * const
gsl::mpi::
gatherVector__name__ = "gatherVector";

const char * const
gsl::mpi::
gatherVector__doc__ = "gather a vector from the members of a communicator";

PyObject *
gsl::mpi::
gatherVector(PyObject *, PyObject * args)
{
    // place holders
    int destination;
    PyObject *communicatorCapsule, *vectorCapsule;

    // parse the argument list
    if (!PyArg_ParseTuple(
                          args,
                          "O!iO!:gatherVector",
                          &PyCapsule_Type, &communicatorCapsule,
                          &destination,
                          &PyCapsule_Type, &vectorCapsule)) {
        return 0;
    }
    // check the communicator capsule
    if (!PyCapsule_IsValid(communicatorCapsule, ::mpi::communicator::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }
    // get the communicator
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>
        (PyCapsule_GetPointer(communicatorCapsule, ::mpi::communicator::capsule_t));

    // check the vector capsule
    if (!PyCapsule_IsValid(vectorCapsule, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "invalid vector capsule for source");
        return 0;
    }
    // get the source vector
    gsl_vector * vector =
        static_cast<gsl_vector *>
        (PyCapsule_GetPointer(vectorCapsule, gsl::vector::capsule_t));

    // the place to deposit the data
    double * data = 0;
    // and the destination vector
    gsl_vector * bertha = 0;

    // at the destination task
    if (comm->rank() == destination) {
        // figure out the shape
        int size = vector->size * comm->size();
        // build the destination vector
        bertha = gsl_vector_alloc(size);
        // and use its payload as the location to deposit the data
        data = bertha->data;
    }

    // gather
    int status = MPI_Gather(
                            vector->data, vector->size, MPI_DOUBLE, // send buffer
                            data, vector->size, MPI_DOUBLE, // receive buffer
                            destination, comm->handle() // address
                            );

    // check the return code
    if (status != MPI_SUCCESS) {
        // and throw an exception if anything went wrong
        PyErr_SetString(PyExc_RuntimeError, "MPI_Gather failed");
        return 0;
    }

    // at all tasks except the destination task
    if (comm->rank() != destination) {
        // return {None}
        Py_INCREF(Py_None);
        return Py_None;
    }

    // wrap the destination vector in a capsule
    PyObject * capsule = PyCapsule_New(bertha, gsl::vector::capsule_t, gsl::vector::free);

    // build the result
    PyObject * result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, capsule);
    PyTuple_SET_ITEM(result, 1, PyLong_FromLong(bertha->size));

    // and return it
    return result;
}


// scatter
const char * const
gsl::mpi::
scatterVector__name__ = "scatterVector";

const char * const
gsl::mpi::
scatterVector__doc__ = "scatter a vector to the members of a communicator";

PyObject *
gsl::mpi::
scatterVector(PyObject *, PyObject * args)
{
    // place holders
    int source;
    PyObject *communicatorCapsule, *vectorCapsule, *destinationCapsule;

    // parse the argument list
    if (!PyArg_ParseTuple(
                          args,
                          "O!iO!O:scatterVector",
                          &PyCapsule_Type, &communicatorCapsule,
                          &source,
                          &PyCapsule_Type, &destinationCapsule,
                          &vectorCapsule // don't force the capsule type check; it may be {None}
                          )) {
        return 0;
    }
    // check the communicator capsule
    if (!PyCapsule_IsValid(communicatorCapsule, ::mpi::communicator::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }
    // check the destination capsule
    if (!PyCapsule_IsValid(destinationCapsule, gsl::vector::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the third argument must be a valid vector");
        return 0;
    }
    // get the communicator
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>
        (PyCapsule_GetPointer(communicatorCapsule, ::mpi::communicator::capsule_t));
    // get the destination vector
    gsl_vector * destination =
        static_cast<gsl_vector *>
        (PyCapsule_GetPointer(destinationCapsule, gsl::vector::capsule_t));

    // the pointer to source payload
    double * data = 0;
    // I only have a valid vector at the {source} rank
    if (comm->rank() == source) {
        // check the vector capsule
        if (!PyCapsule_IsValid(vectorCapsule, gsl::vector::capsule_t)) {
            PyErr_SetString(PyExc_TypeError, "invalid vector capsule for source");
            return 0;
        }
        // get the source vector
        gsl_vector * vector =
            static_cast<gsl_vector *>
            (PyCapsule_GetPointer(vectorCapsule, gsl::vector::capsule_t));
        // and extract the pointer to the payload
        data = vector->data;
    }

    // get the length of the destination vector
    int length = destination->size;
    // scatter the data
    int status = MPI_Scatter(
                         data, length, MPI_DOUBLE, // source buffer
                         destination->data, length, MPI_DOUBLE, // destination buffer
                         source, comm->handle() // address
                         );

    // check the return code
    if (status != MPI_SUCCESS) {
        // and throw an exception if anything went wrong
        PyErr_SetString(PyExc_RuntimeError, "MPI_Scatter failed");
        return 0;
    }

    // return
    Py_INCREF(Py_None);
    return Py_None;
}


// end of file
