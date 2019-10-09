// -*- C++ -*-
//
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#include <portinfo>
#include <Python.h>
#include <pyre/mpi.h>

#include <pyre/journal.h>

#include "capsules.h"
#include "groups.h"
#include "exceptions.h"

// check whether the given group is empty
const char * const mpi::group::isEmpty__name__ = "groupIsEmpty";
const char * const mpi::group::isEmpty__doc__ = "check whether the given group is empty";

PyObject * mpi::group::isEmpty(PyObject *, PyObject * args)
{
    // placeholder for the python object
    PyObject * py_group;

    // extract the communicator from the argument tuple
    if (!PyArg_ParseTuple(args, "O!:groupIsEmpty", &PyCapsule_Type, &py_group)) {
        return 0;
    }

    // check that we were handed the correct kind of capsule
    if (!PyCapsule_IsValid(py_group, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid group");
        return 0;
    }

    // get the group
    pyre::mpi::group_t * group =
        static_cast<pyre::mpi::group_t *>(PyCapsule_GetPointer(py_group, capsule_t));

    // check and return
    return PyBool_FromLong(group->isEmpty());
}


// create a communicator group (MPI_Comm_group)
const char * const mpi::group::create__name__ = "groupCreate";
const char * const mpi::group::create__doc__ = "create a communicator group";

PyObject * mpi::group::create(PyObject *, PyObject * args)
{
    // placeholder for the python object
    PyObject * py_comm;

    // extract the communicator from the argument tuple
    if (!PyArg_ParseTuple(args, "O!:groupCreate", &PyCapsule_Type, &py_comm)) {
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

    // build the associated group
    pyre::mpi::group_t * group = new pyre::mpi::group_t(comm->group());

    if (!group) {
        PyErr_SetString(PyExc_ValueError, "group could not be created");
        return 0;
    }

    // wrap in a capsule and return the new communicator
    return PyCapsule_New(group, capsule_t, free);
}


// return the communicator group size (MPI_Group_size)
const char * const mpi::group::size__name__ = "groupSize";
const char * const mpi::group::size__doc__ = "retrieve the group size";

PyObject * mpi::group::size(PyObject *, PyObject * args)
{
    // placeholder
    PyObject * py_group;

    // parse the argument list
    if (!PyArg_ParseTuple(args, "O!:groupSize", &PyCapsule_Type, &py_group)) {
        return 0;
    }

    // check that we were handed the correct kind of capsule
    if (!PyCapsule_IsValid(py_group, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid group");
        return 0;
    }

    // get the group
    pyre::mpi::group_t * group =
        static_cast<pyre::mpi::group_t *>(PyCapsule_GetPointer(py_group, capsule_t));

    // extract the group size and return it
    return PyLong_FromLong(group->size());
}


// return the process rank in a given communicator group (MPI_Group_rank)
const char * const mpi::group::rank__name__ = "groupRank";
const char * const mpi::group::rank__doc__ = "retrieve the rank of this process";

PyObject * mpi::group::rank(PyObject *, PyObject * args)
{
    // placeholder
    PyObject * py_group;

    // parse the argument list
    if (!PyArg_ParseTuple(args, "O!:groupSize", &PyCapsule_Type, &py_group)) {
        return 0;
    }

    // check that we were handed the correct kind of capsule
    if (!PyCapsule_IsValid(py_group, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid group");
        return 0;
    }

    // get the group
    pyre::mpi::group_t * group =
        static_cast<pyre::mpi::group_t *>(PyCapsule_GetPointer(py_group, capsule_t));

    // extract the group size and return it
    return PyLong_FromLong(group->rank());
}


// return the process rank in a given communicator group (MPI_Group_incl)
const char * const mpi::group::include__name__ = "groupInclude";
const char * const mpi::group::include__doc__ = "include processors in this group";

PyObject * mpi::group::include(PyObject *, PyObject * args)
{
    PyObject * py_group;
    PyObject * rankSeq;

    if (!PyArg_ParseTuple(
                          args,
                          "O!O:groupInclude",
                          &PyCapsule_Type, &py_group,
                          &rankSeq)) {
        return 0;
    }
    // check that we were handed the correct kind of capsule
    if (!PyCapsule_IsValid(py_group, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid group");
        return 0;
    }
    // check the rank sequence
    if (!PySequence_Check(rankSeq)) {
        PyErr_SetString(PyExc_TypeError, "the second argument must be a sequence");
        return 0;
    }

    // get the communicator group
    pyre::mpi::group_t * group =
        static_cast<pyre::mpi::group_t *>(PyCapsule_GetPointer(py_group, capsule_t));

    // store the ranks in a vector
    int size = PySequence_Length(rankSeq);
    pyre::mpi::group_t::ranklist_t ranks;
    for (int i = 0; i < size; ++i) {
        ranks.push_back(PyLong_AsLong(PySequence_GetItem(rankSeq, i)));
    }

    // make the MPI call
    pyre::mpi::group_t * newGroup = new pyre::mpi::group_t(group->include(ranks));

    // otherwise, wrap it in a capsule and return it
    return PyCapsule_New(newGroup, capsule_t, free);
}


// return the process rank in a given communicator group (MPI_Group_excl)
const char * const mpi::group::exclude__name__ = "groupExclude";
const char * const mpi::group::exclude__doc__ = "exclude processors from this group";

PyObject * mpi::group::exclude(PyObject *, PyObject * args)
{
    PyObject * py_group;
    PyObject * rankSeq;

    if (!PyArg_ParseTuple(
                          args,
                          "O!O:groupExclude",
                          &PyCapsule_Type, &py_group,
                          &rankSeq)) {
        return 0;
    }
    // check that we were handed the correct kind of capsule
    if (!PyCapsule_IsValid(py_group, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid group");
        return 0;
    }
    // check the rank sequence
    if (!PySequence_Check(rankSeq)) {
        PyErr_SetString(PyExc_TypeError, "the second argument must be a sequence");
        return 0;
    }

    // get the communicator group
    pyre::mpi::group_t * group =
        static_cast<pyre::mpi::group_t *>(PyCapsule_GetPointer(py_group, capsule_t));

    // store the ranks in a vector
    int size = PySequence_Length(rankSeq);
    pyre::mpi::group_t::ranklist_t ranks;
    for (int i = 0; i < size; ++i) {
        ranks.push_back(PyLong_AsLong(PySequence_GetItem(rankSeq, i)));
    }

    // make the MPI call
    pyre::mpi::group_t * newGroup = new pyre::mpi::group_t(group->exclude(ranks));

    // wrap into a capsule and return
    return PyCapsule_New(newGroup, capsule_t, free);
}


// build a group out of the union of two others
const char * const mpi::group::add__name__ = "groupUnion";
const char * const mpi::group::add__doc__ = "build a group out of the union of two others";

PyObject * mpi::group::add(PyObject *, PyObject * args)
{
    PyObject * py_g1;
    PyObject * py_g2;

    if (!PyArg_ParseTuple(
                          args,
                          "O!O!:groupUnion",
                          &PyCapsule_Type, &py_g1,
                          &PyCapsule_Type, &py_g2)) {
        return 0;
    }
    // check that we were handed the correct kind of capsules
    if (!PyCapsule_IsValid(py_g1, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid group");
        return 0;
    }
    if (!PyCapsule_IsValid(py_g2, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the second argument must be a valid group");
        return 0;
    }

    // get the communicator groups
    pyre::mpi::group_t * g1 =
        static_cast<pyre::mpi::group_t *>(PyCapsule_GetPointer(py_g1, capsule_t));
    pyre::mpi::group_t * g2 =
        static_cast<pyre::mpi::group_t *>(PyCapsule_GetPointer(py_g2, capsule_t));


    // make the MPI call
    pyre::mpi::group_t * newGroup = new pyre::mpi::group_t(pyre::mpi::groupUnion(*g1, *g2));

    // wrap into a capsule and return
    return PyCapsule_New(newGroup, capsule_t, free);
}


// build a group out of the intersection of two others
const char * const mpi::group::intersect__name__ = "groupIntersection";
const char * const mpi::group::intersect__doc__ =
    "build a group out of the intersection of two others";

PyObject * mpi::group::intersect(PyObject *, PyObject * args)
{
    PyObject * py_g1;
    PyObject * py_g2;

    if (!PyArg_ParseTuple(
                          args,
                          "O!O!:groupIntersection",
                          &PyCapsule_Type, &py_g1,
                          &PyCapsule_Type, &py_g2)) {
        return 0;
    }
    // check that we were handed the correct kind of capsules
    if (!PyCapsule_IsValid(py_g1, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid group");
        return 0;
    }
    if (!PyCapsule_IsValid(py_g2, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the second argument must be a valid group");
        return 0;
    }

    // get the communicator groups
    pyre::mpi::group_t * g1 =
        static_cast<pyre::mpi::group_t *>(PyCapsule_GetPointer(py_g1, capsule_t));
    pyre::mpi::group_t * g2 =
        static_cast<pyre::mpi::group_t *>(PyCapsule_GetPointer(py_g2, capsule_t));


    // make the MPI call
    pyre::mpi::group_t * newGroup = new pyre::mpi::group_t(pyre::mpi::groupIntersection(*g1, *g2));

    // wrap into a capsule and return
    return PyCapsule_New(newGroup, capsule_t, free);
}


// build a group out of the difference of two others
const char * const mpi::group::subtract__name__ = "groupDifference";
const char * const mpi::group::subtract__doc__ =
    "build a group out of the difference of two others";

PyObject * mpi::group::subtract(PyObject *, PyObject * args)
{
    PyObject * py_g1;
    PyObject * py_g2;

    if (!PyArg_ParseTuple(
                          args,
                          "O!O!:groupDifference",
                          &PyCapsule_Type, &py_g1,
                          &PyCapsule_Type, &py_g2)) {
        return 0;
    }
    // check that we were handed the correct kind of capsules
    if (!PyCapsule_IsValid(py_g1, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the first argument must be a valid group");
        return 0;
    }
    if (!PyCapsule_IsValid(py_g2, capsule_t)) {
        PyErr_SetString(PyExc_TypeError, "the second argument must be a valid group");
        return 0;
    }

    // get the communicator groups
    pyre::mpi::group_t * g1 =
        static_cast<pyre::mpi::group_t *>(PyCapsule_GetPointer(py_g1, capsule_t));
    pyre::mpi::group_t * g2 =
        static_cast<pyre::mpi::group_t *>(PyCapsule_GetPointer(py_g2, capsule_t));

    // make the MPI call
    pyre::mpi::group_t * newGroup = new pyre::mpi::group_t(pyre::mpi::groupDifference(*g1, *g2));

    // wrap into a capsule and return
    return PyCapsule_New(newGroup, capsule_t, free);
}


// helpers
void
mpi::group::
free(PyObject * py_group)
{
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(py_group, capsule_t)) {
        return;
    }
    // get the pointer
    pyre::mpi::group_t * group =
        static_cast<pyre::mpi::group_t *>(PyCapsule_GetPointer(py_group, capsule_t));

    pyre::journal::debug_t info("mpi.fini");
    info
        << pyre::journal::at(__HERE__)
        << "group@" << group << ": deleting"
        << pyre::journal::endl;

    // delete it
    delete group;
    // and return
    return;
}

// end of file
