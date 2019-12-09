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
#include "ports.h"
#include "exceptions.h"

// send bytes
const char * const mpi::port::sendBytes__name__ = "sendBytes";
const char * const mpi::port::sendBytes__doc__ = "send bytes to a peer";

PyObject * mpi::port::sendBytes(PyObject *, PyObject * args)
{
    // placeholder for the arguments
    int tag;
    int peer;
    char * str;
    Py_ssize_t len;
    PyObject * py_comm;

    // extract the arguments from the tuple
    if (!PyArg_ParseTuple(
                          args,
                          "O!iiy#:sendBytes",
                          &PyCapsule_Type, &py_comm,
                          &peer, &tag,
                          &str, &len)) {
        return 0;
    }

    // check that we were handed the correct kind of communicator capsule
    if (!PyCapsule_IsValid(py_comm, mpi::communicator::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }

    // convert into the pyre::mpi object
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>
        (PyCapsule_GetPointer(py_comm, mpi::communicator::capsule_t));

    // dump arguments
    pyre::journal::debug_t info("mpi.ports");
    info
        << pyre::journal::at(__HERE__)
        << "peer={" << peer
        << "}, tag={" << tag
        << "}, bytes={" << len << "} at " << (void *)str
        << pyre::journal::endl;

    // send the data
    MPI_Send(str, len, MPI_BYTE, peer, tag, comm->handle());

    // return
    Py_INCREF(Py_None);
    return Py_None;
}


// receive bytes
const char * const mpi::port::recvBytes__name__ = "recvBytes";
const char * const mpi::port::recvBytes__doc__ = "receive bytes from a peer";

PyObject * mpi::port::recvBytes(PyObject *, PyObject * args)
{
    // placeholders for the arguments
    int tag;
    int peer;
    PyObject * py_comm;

    // extract the arguments from the tuple
    if (!PyArg_ParseTuple(
                          args,
                          "O!ii:recvBytes",
                          &PyCapsule_Type, &py_comm,
                          &peer, &tag)) {
        return 0;
    }

    // check that we were handed the correct kind of communicator capsule
    if (!PyCapsule_IsValid(py_comm, mpi::communicator::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }

    // convert into the pyre::mpi object
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>
        (PyCapsule_GetPointer(py_comm, mpi::communicator::capsule_t));

    int len;
    char * str;
    // block until an appropriate message has arrived
    MPI_Status status;
    MPI_Probe(peer, tag, comm->handle(), &status);
    // get the size of the message, including the terminating null
    MPI_Get_count(&status, MPI_BYTE, &len);
    // allocate a buffer
    str = new char[len];
    // receive the data
    MPI_Recv(str, len, MPI_BYTE, peer, tag, comm->handle(), &status);

    // dump message
    pyre::journal::debug_t info("mpi.ports");
    info
        << pyre::journal::at(__HERE__)
        << "peer={" << peer
        << "}, tag={" << tag
        << "}, bytes={" << len << "} at " << (void *)str
        << pyre::journal::endl;

    // build the return value
    PyObject * value = Py_BuildValue("y#", str, len);

    // clean up
    delete [] str;

    // return
    return value;
}

// send a string
const char * const mpi::port::sendString__name__ = "sendString";
const char * const mpi::port::sendString__doc__ = "send a string to a peer";

PyObject * mpi::port::sendString(PyObject *, PyObject * args)
{
    // placeholder for the arguments
    int tag;
    int peer;
    char * str;
    Py_ssize_t len;
    PyObject * py_comm;

    // extract the arguments from the tuple
    if (!PyArg_ParseTuple(
                          args,
                          "O!iis#:sendString",
                          &PyCapsule_Type, &py_comm,
                          &peer, &tag,
                          &str, &len)) {
        return 0;
    }

    // check that we were handed the correct kind of communicator capsule
    if (!PyCapsule_IsValid(py_comm, mpi::communicator::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }

    // convert into the pyre::mpi object
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>
        (PyCapsule_GetPointer(py_comm, mpi::communicator::capsule_t));

    // dump arguments
    pyre::journal::debug_t info("mpi.ports");
    info
        << pyre::journal::at(__HERE__)
        << "peer={" << peer
        << "}, tag={" << tag
        << "}, string={" << str << "}@" << len
        << pyre::journal::endl;

    // send the data (along with the terminating null)
    MPI_Send(str, len+1, MPI_CHAR, peer, tag, comm->handle());

    // return
    Py_INCREF(Py_None);
    return Py_None;
}


// receive a string
const char * const mpi::port::recvString__name__ = "recvString";
const char * const mpi::port::recvString__doc__ = "receive a string from a peer";

PyObject * mpi::port::recvString(PyObject *, PyObject * args)
{
    // placeholders for the arguments
    int tag;
    int peer;
    PyObject * py_comm;

    // extract the arguments from the tuple
    if (!PyArg_ParseTuple(
                          args,
                          "O!ii:recvString",
                          &PyCapsule_Type, &py_comm,
                          &peer, &tag)) {
        return 0;
    }

    // check that we were handed the correct kind of communicator capsule
    if (!PyCapsule_IsValid(py_comm, mpi::communicator::capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid communicator");
        return 0;
    }

    // convert into the pyre::mpi object
    pyre::mpi::communicator_t * comm =
        static_cast<pyre::mpi::communicator_t *>
        (PyCapsule_GetPointer(py_comm, mpi::communicator::capsule_t));

    int len;
    char * str;
    // block until an appropriate message has arrived
    MPI_Status status;
    MPI_Probe(peer, tag, comm->handle(), &status);
    // get the size of the message, including the terminating null
    MPI_Get_count(&status, MPI_CHAR, &len);
    // allocate a buffer
    str = new char[len];
    // receive the data
    MPI_Recv(str, len, MPI_CHAR, peer, tag, comm->handle(), &status);

    // dump message
    pyre::journal::debug_t info("mpi.ports");
    info
        << pyre::journal::at(__HERE__)
        << "peer={" << peer
        << "}, tag={" << tag
        << "}, string={" << str << "}@" << len
        << pyre::journal::endl;

    // build the return value
    PyObject * value = Py_BuildValue("s", str);

    // clean up
    delete [] str;

    // return
    return value;
}

// end of file
