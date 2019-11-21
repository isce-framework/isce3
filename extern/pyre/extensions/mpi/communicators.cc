// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#include <portinfo>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <pyre/mpi.h>
#include <pyre/journal.h>

#include "capsules.h"
#include "communicators.h"
#include "exceptions.h"


// the predefined groups
PyObject *
mpi::communicator::
world = PyCapsule_New(new pyre::mpi::communicator_t(MPI_COMM_WORLD), capsule_t, 0);

// create a communicator (MPI_Comm_create)
const char * const mpi::communicator::create__name__ = "communicatorCreate";
const char * const mpi::communicator::create__doc__ = "create a communicator";

PyObject * mpi::communicator::create(PyObject *, PyObject * args)
{
    // placeholders for the python objects
    PyObject * py_old;
    PyObject * py_group;

    // extract them from the argument tuple in a type safe manner
    if (!PyArg_ParseTuple(
                          args,
                          "O!O!:communicatorCreate",
                          &PyCapsule_Type, &py_old, &PyCapsule_Type, &py_group)) {
        return 0;
    }
    // check that we were handed the correct kind of communicator capsule
    if (!PyCapsule_IsValid(py_old, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }
    // check that we were handed the correct kind of group capsule
    if (!PyCapsule_IsValid(py_group, mpi::group::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the second argument must be a valid communicator group");
        return 0;
    }

    // convert into the pyre::mpi objects
    pyre::mpi::communicator_t * old =
        static_cast<pyre::mpi::communicator_t *>(PyCapsule_GetPointer(py_old, capsule_t));
    pyre::mpi::group_t * group =
        static_cast<pyre::mpi::group_t *>(PyCapsule_GetPointer(py_group, mpi::group::capsule_t));

    // create the new communicator
    pyre::mpi::communicator_t * comm = new pyre::mpi::communicator_t(old->communicator(*group));

    // if the creation failed
    if (comm->handle() == MPI_COMM_NULL) {
        // bail out
        Py_INCREF(Py_None);
        return Py_None;
    }

    // otherwise, wrap the handle in a capsule and return it
    return PyCapsule_New(comm, capsule_t, free);
}

// create a cartesian communicator (MPI_Cart_create)
const char * const mpi::cartesian::create__name__ = "communicatorCreateCartesian";
const char * const mpi::cartesian::create__doc__ = "create a Cartesian communicator";

PyObject * mpi::cartesian::create(PyObject *, PyObject * args)
{
    // placeholders for the argument list
    int reorder;
    PyObject * py_comm;
    PyObject * procSeq;
    PyObject * periodSeq;

    pyre::journal::debug_t info("mpi.cartesian");

    // extract them from the argument tuple
    if (!PyArg_ParseTuple(
                          args,
                          "O!iOO:communicatorCreateCartesian",
                          &PyCapsule_Type, &py_comm,
                          &reorder, &procSeq, &periodSeq)) {
        return 0;
    }

    // check that we were handed the correct kind of capsule
    if (!PyCapsule_IsValid(py_comm, mpi::communicator::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }
    // check the processor sequence
    if (!PySequence_Check(procSeq)) {
        PyErr_SetString(PyExc_TypeError, "the third argument must be a sequence");
        return 0;
    }
    // check the period sequence
    if (!PySequence_Check(periodSeq)) {
        PyErr_SetString(PyExc_TypeError, "the fourth argument must be a sequence");
        return 0;
    }

    // get the communicator
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>
        (PyCapsule_GetPointer(py_comm, mpi::communicator::capsule_t));

    // compute the dimensionality of the communicator
    int size = PySequence_Size(procSeq);
    if (size != PySequence_Size(periodSeq)) {
        PyErr_SetString(Error, "mismatch in size of processor and period lists");
        return 0;
    }

    info << pyre::journal::at(__HERE__) << "dimension = " << size << pyre::journal::newline;

    // allocate the vectors
    std::vector<int> procs;
    std::vector<int> periods;

    // copy the data over
    info << pyre::journal::at(__HERE__) << "axes: ";

    for (int dim = 0; dim < size; ++dim) {
        procs.push_back(PyLong_AsLong(PySequence_GetItem(procSeq, dim)));
        periods.push_back(PyLong_AsLong(PySequence_GetItem(periodSeq, dim)));

        info << " (" << procs[dim] << "," << periods[dim] << ")";

    }

    info << pyre::journal::endl;

    // make the MPI call
    pyre::mpi::communicator_t * cartesian =
        new pyre::mpi::communicator_t(comm->cartesian(procs, periods, reorder));

    info
        << pyre::journal::at(__HERE__)
        << "created cartesian@" << cartesian << " from comm@" << comm
        << pyre::journal::endl;

// clean up and return
    if (!cartesian) {
        PyErr_SetString(Error, "could not build cartesian communicator");
        return 0;
    }

    // return the new communicator
    return PyCapsule_New(cartesian, mpi::communicator::capsule_t, mpi::communicator::free);
}


// return the communicator size (MPI_Comm_size)
const char * const mpi::communicator::size__name__ = "communicatorSize";
const char * const mpi::communicator::size__doc__ = "get the size of a communicator";

PyObject * mpi::communicator::size(PyObject *, PyObject * args)
{
    // placeholder
    PyObject * py_comm;

    // parse the argument list
    if (!PyArg_ParseTuple(args, "O!:communicatorSize", &PyCapsule_Type, &py_comm)) {
        return 0;
    }

    // check that we were handed the correct kind of capsule
    if (!PyCapsule_IsValid(py_comm, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }

    // get the communicator
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>(PyCapsule_GetPointer(py_comm, capsule_t));

    // extract the communicator size and return it
    return PyLong_FromLong(comm->size());
}


// return the communicator rank (MPI_Comm_rank)
const char * const mpi::communicator::rank__name__ = "communicatorRank";
const char * const mpi::communicator::
rank__doc__ = "get the rank of this process in the given communicator";

PyObject * mpi::communicator::rank(PyObject *, PyObject * args)
{
    // placeholder
    PyObject * py_comm;

    // parse the argument list
    if (!PyArg_ParseTuple(args, "O!:communicatorRank", &PyCapsule_Type, &py_comm)) {
        return 0;
    }

    // check that we were handed the correct kind of capsule
    if (!PyCapsule_IsValid(py_comm, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }

    // get the communicator
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>(PyCapsule_GetPointer(py_comm, capsule_t));

    // return
    return PyLong_FromLong(comm->rank());
}


// set a communicator barrier (MPI_Barrier)
const char * const mpi::communicator::barrier__name__ = "communicatorBarrier";
const char * const mpi::communicator::
barrier__doc__ = "block until all members of this communicator reach this point";

PyObject * mpi::communicator::barrier(PyObject *, PyObject * args)
{
    // placeholder
    PyObject * py_comm;

    // parse the argument list
    if (!PyArg_ParseTuple(args, "O!:communicatorBarrier", &PyCapsule_Type, &py_comm)) {
        return 0;
    }

    // check that we were handed the correct kind of capsule
    if (!PyCapsule_IsValid(py_comm, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }

    // get the communicator
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>(PyCapsule_GetPointer(py_comm, capsule_t));

    // set up the barrier
    comm->barrier();

    // and return
    Py_INCREF(Py_None);
    return Py_None;

}


// return the coordinates of the process in the cartesian communicator (MPI_Cart_coords)
const char * const
mpi::cartesian::
coordinates__name__ = "communicatorCartesianCoordinates";

const char * const
mpi::cartesian::
coordinates__doc__ = "retrieve the cartesian coordinates of this process";

PyObject * mpi::cartesian::coordinates(PyObject *, PyObject * args)
{
    // placeholders
    int dim;
    int rank;
    PyObject * py_comm;

    // parse the argument list
    if (!PyArg_ParseTuple(
                          args,
                          "O!ii:communicatorCartesianCoordinates",
                          &PyCapsule_Type, &py_comm,
                          &rank, &dim)) {
        return 0;
    }

    // check that we were handed the correct kind of capsule
    if (!PyCapsule_IsValid(py_comm, mpi::communicator::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }

    pyre::mpi::communicator_t::ranklist_t coordinates;
    // get the communicator
    pyre::mpi::communicator_t * cartesian =
        static_cast<pyre::mpi::communicator_t *>
        (PyCapsule_GetPointer(py_comm, mpi::communicator::capsule_t));

    // dump
    pyre::journal::debug_t info("mpi.cartesian");
    if (info.isActive()) {
        int wr, ws;
        MPI_Comm_rank(MPI_COMM_WORLD, &wr);
        MPI_Comm_size(MPI_COMM_WORLD, &ws);
        info
            << pyre::journal::at(__HERE__)
            << "[" << wr << ":" << ws << "] "
            << "communicator@" << cartesian << ": "
            << dim << "-dim cartesian communicator, rank=" << rank
            << pyre::journal::newline;
    }

    coordinates = cartesian->coordinates(rank);
    info << "coordinates:";
    for (int i=0; i < dim; ++i) {
        info << " " << coordinates[i];
    }
    info << pyre::journal::endl;

    PyObject *value = PyTuple_New(dim);
    for (int i = 0; i < dim; ++i) {
        PyTuple_SET_ITEM(value, i, PyLong_FromLong(coordinates[i]));
    }

    // and return
    return value;
}


// broadcast a python object to all tasks
const char * const
mpi::communicator::
bcast__name__ = "bcast";

const char * const
mpi::communicator::
bcast__doc__ = "broadcast a python object to all tasks";

PyObject * mpi::communicator::bcast(PyObject *, PyObject * args)
{
    // place holders
    int rank, root;
    char * data;
    Py_ssize_t len;
    PyObject * py_comm;

    // parse the argument list
    if (!PyArg_ParseTuple(
                          args,
                          "O!iiy#:bcast",
                          &PyCapsule_Type, &py_comm,
                          &rank, &root,
                          &data, &len)) {
        return 0;
    }

    // check that we were handed the correct kind of capsule
    if (!PyCapsule_IsValid(py_comm, mpi::communicator::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }
    // get the communicator
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>
        (PyCapsule_GetPointer(py_comm, mpi::communicator::capsule_t));

    // trying to stay Py_ssize_t clean...
    int size = len;
    // broadcast the length of the data buffer
    MPI_Bcast(&size, 1, MPI_INT, root, comm->handle());
    // trying to stay Py_ssize_t clean...
    len = size;
    // everybody except {root}
    if (rank != root) {
        // must allocate space to receive the data
        data = new char[len];
    }
    // broadcast the data
    MPI_Bcast(data, len, MPI_BYTE, root, comm->handle());

    // build the return value
    PyObject * value = Py_BuildValue("y#", data, len);
    // everybody except {root}
    if (rank != root) {
        // must clean up
        delete [] data;
    }
    // all done
    return value;
}


// perform a sum reduction
const char * const
mpi::communicator::
sum__name__ = "sum";

const char * const
mpi::communicator::
sum__doc__ = "perform a sum reduction";

PyObject * mpi::communicator::sum(PyObject *, PyObject * args)
{
    // place holders
    int root;
    double number;
    PyObject * py_comm;

    // parse the argument list
    if (!PyArg_ParseTuple(args, "O!id:sum", &PyCapsule_Type, &py_comm, &root, &number)) {
        return 0;
    }

    // check that we were handed the correct kind of capsule
    if (!PyCapsule_IsValid(py_comm, mpi::communicator::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }
    // get the communicator
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>
        (PyCapsule_GetPointer(py_comm, mpi::communicator::capsule_t));

    // space for the result
    double total;
    // compute the total
    MPI_Reduce(&number, &total, 1, MPI_DOUBLE, MPI_SUM, root, comm->handle());

    // at {root}
    if (comm->rank() == root) {
        // return the reduced value
        return PyFloat_FromDouble(total);
    }
    // everybody else gets {None}
    Py_INCREF(Py_None);
    return Py_None;
}


// perform a product reduction
const char * const
mpi::communicator::
product__name__ = "product";

const char * const
mpi::communicator::
product__doc__ = "perform a product reduction";

PyObject * mpi::communicator::product(PyObject *, PyObject * args)
{
    // place holders
    int root;
    double number;
    PyObject * py_comm;

    // parse the argument list
    if (!PyArg_ParseTuple(args, "O!id:product", &PyCapsule_Type, &py_comm, &root, &number)) {
        return 0;
    }

    // check that we were handed the correct kind of capsule
    if (!PyCapsule_IsValid(py_comm, mpi::communicator::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }
    // get the communicator
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>
        (PyCapsule_GetPointer(py_comm, mpi::communicator::capsule_t));

    // space for the result
    double product = 0;
    // compute the product
    MPI_Reduce(&number, &product, 1, MPI_DOUBLE, MPI_PROD, root, comm->handle());

    // at {root}
    if (comm->rank() == root) {
        // return the reduced value
        return PyFloat_FromDouble(product);
    }
    // everybody else gets {None}
    Py_INCREF(Py_None);
    return Py_None;
}


// perform a max reduction
const char * const
mpi::communicator::
max__name__ = "max";

const char * const
mpi::communicator::
max__doc__ = "perform a max reduction";

PyObject * mpi::communicator::max(PyObject *, PyObject * args)
{
    // place holders
    int root;
    double number;
    PyObject * py_comm;

    // parse the argument list
    if (!PyArg_ParseTuple(args, "O!id:max", &PyCapsule_Type, &py_comm, &root, &number)) {
        return 0;
    }

    // check that we were handed the correct kind of capsule
    if (!PyCapsule_IsValid(py_comm, mpi::communicator::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }

    // get the communicator
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>
        (PyCapsule_GetPointer(py_comm, mpi::communicator::capsule_t));

    // space for the result
    double largest = 0;
    // compute the total
    MPI_Reduce(&number, &largest, 1, MPI_DOUBLE, MPI_MAX, root, comm->handle());

    // at {root}
    if (comm->rank() == root) {
        // return the reduced value
        return PyFloat_FromDouble(largest);
    }
    // everybody else gets {None}
    Py_INCREF(Py_None);
    return Py_None;
}


// perform a min reduction
const char * const
mpi::communicator::
min__name__ = "min";

const char * const
mpi::communicator::
min__doc__ = "perform a min reduction";

PyObject * mpi::communicator::min(PyObject *, PyObject * args)
{
    // place holders
    int root;
    double number;
    PyObject * py_comm;

    // parse the argument list
    if (!PyArg_ParseTuple(args, "O!id:min", &PyCapsule_Type, &py_comm, &root, &number)) {
        return 0;
    }

    // check that we were handed the correct kind of capsule
    if (!PyCapsule_IsValid(py_comm, mpi::communicator::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }
    // get the communicator
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>
        (PyCapsule_GetPointer(py_comm, mpi::communicator::capsule_t));

    // space for the result
    double smallest = 0;
    // compute the smallest
    MPI_Reduce(&number, &smallest, 1, MPI_DOUBLE, MPI_MIN, root, comm->handle());

    // at {root}
    if (comm->rank() == root) {
        // return the reduced value
        return PyFloat_FromDouble(smallest);
    }
    // everybody else gets {None}
    Py_INCREF(Py_None);
    return Py_None;
}

// perform a sum reduction and distribute the result back to all processes
const char * const
mpi::communicator::
sum_all__name__ = "sum_all";

const char * const
mpi::communicator::
sum_all__doc__ = "perform a sum reduction and distribute the result back to all processes";

PyObject * mpi::communicator::sum_all(PyObject *, PyObject * args)
{
    // place holders
    double number;
    PyObject * py_comm;

    // parse the argument list
    if (!PyArg_ParseTuple(args, "O!d:sum_all", &PyCapsule_Type, &py_comm, &number)) {
        return 0;
    }

    // check that we were handed the correct kind of capsule
    if (!PyCapsule_IsValid(py_comm, mpi::communicator::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }
    // get the communicator
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>
        (PyCapsule_GetPointer(py_comm, mpi::communicator::capsule_t));

    // space for the result
    double total;
    // compute the total
    MPI_Allreduce(&number, &total, 1, MPI_DOUBLE, MPI_SUM, comm->handle());

    // return the reduced value for all processes
    return PyFloat_FromDouble(total);
}


// perform a product reduction and distribute the result back to all processes
const char * const
mpi::communicator::
product_all__name__ = "product_all";

const char * const
mpi::communicator::
product_all__doc__ = "perform a product reduction and distribute the result back to all processes";

PyObject * mpi::communicator::product_all(PyObject *, PyObject * args)
{
    // place holders
    double number;
    PyObject * py_comm;

    // parse the argument list
    if (!PyArg_ParseTuple(args, "O!d:product_all", &PyCapsule_Type, &py_comm, &number)) {
        return 0;
    }

    // check that we were handed the correct kind of capsule
    if (!PyCapsule_IsValid(py_comm, mpi::communicator::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }
    // get the communicator
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>
        (PyCapsule_GetPointer(py_comm, mpi::communicator::capsule_t));

    // space for the result
    double product = 0;
    // compute the product
    MPI_Allreduce(&number, &product, 1, MPI_DOUBLE, MPI_PROD, comm->handle());

    // return the reduced value for all processes
    return PyFloat_FromDouble(product);
}


// perform a max reduction and distribute the result back to all processes
const char * const
mpi::communicator::
max_all__name__ = "max_all";

const char * const
mpi::communicator::
max_all__doc__ = "perform a max reduction and distribute the result back to all processes";

PyObject * mpi::communicator::max_all(PyObject *, PyObject * args)
{
    // place holders
    double number;
    PyObject * py_comm;

    // parse the argument list
    if (!PyArg_ParseTuple(args, "O!d:max_all", &PyCapsule_Type, &py_comm, &number)) {
        return 0;
    }

    // check that we were handed the correct kind of capsule
    if (!PyCapsule_IsValid(py_comm, mpi::communicator::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }

    // get the communicator
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>
        (PyCapsule_GetPointer(py_comm, mpi::communicator::capsule_t));

    // space for the result
    double largest = 0;
    // compute the total
    MPI_Allreduce(&number, &largest, 1, MPI_DOUBLE, MPI_MAX, comm->handle());

    // return the reduced value
    return PyFloat_FromDouble(largest);
}


// perform a min reduction and distribute the result back to all processes
const char * const
mpi::communicator::
min_all__name__ = "min_all";

const char * const
mpi::communicator::
min_all__doc__ = "perform a min reduction and distribute the result back to all processes";

PyObject * mpi::communicator::min_all(PyObject *, PyObject * args)
{
    // place holders
    double number;
    PyObject * py_comm;

    // parse the argument list
    if (!PyArg_ParseTuple(args, "O!d:min_all", &PyCapsule_Type, &py_comm, &number)) {
        return 0;
    }

    // check that we were handed the correct kind of capsule
    if (!PyCapsule_IsValid(py_comm, mpi::communicator::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }
    // get the communicator
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>
        (PyCapsule_GetPointer(py_comm, mpi::communicator::capsule_t));

    // space for the result
    double smallest = 0;
    // compute the smallest
    MPI_Allreduce(&number, &smallest, 1, MPI_DOUBLE, MPI_MIN, comm->handle());

    // return the reduced value
    return PyFloat_FromDouble(smallest);
}

// helpers
void
mpi::communicator::
free(PyObject * capsule)
{
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        return;
    }
    // get the pointer
    pyre::mpi::communicator_t * communicator =
        static_cast<pyre::mpi::communicator_t *>(PyCapsule_GetPointer(capsule, capsule_t));

    // generate a diagnostic
    pyre::journal::debug_t info("mpi.fini");
    info
        << pyre::journal::at(__HERE__)
        << "[" << communicator->rank() << ":" << communicator->size() << "] "
        << "deleting comm@" << communicator
        << pyre::journal::endl;

    // delete the communicator
    delete communicator;
    // all done
    return;
}

// end of file
