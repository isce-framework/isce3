// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(pyre_extensions_mpi_groups_h)
#define pyre_extensions_mpi_groups_h

// place everything in my private namespace
namespace mpi {
    namespace group {

        // the predefined groups
        extern PyObject * null;
        extern PyObject * empty;

        // check whether a group is empty
        extern const char * const isEmpty__name__;
        extern const char * const isEmpty__doc__;
        PyObject * isEmpty(PyObject *, PyObject *);

        // create a communicator group (MPI_Comm_group)
        extern const char * const create__name__;
        extern const char * const create__doc__;
        PyObject * create(PyObject *, PyObject *);

        // return the communicator group size (MPI_Group_size)
        extern const char * const size__name__;
        extern const char * const size__doc__;
        PyObject * size(PyObject *, PyObject *);

        // return the process rank in a given communicator group (MPI_Group_rank)
        extern const char * const rank__name__;
        extern const char * const rank__doc__;
        PyObject * rank(PyObject *, PyObject *);

        // return the process rank in a given communicator group (MPI_Group_incl)
        extern const char * const include__name__;
        extern const char * const include__doc__;
        PyObject * include(PyObject *, PyObject *);

        // return the process rank in a given communicator group (MPI_Group_excl)
        extern const char * const exclude__name__;
        extern const char * const exclude__doc__;
        PyObject * exclude(PyObject *, PyObject *);

        // build a group out of the union of two others
        extern const char * const add__name__;
        extern const char * const add__doc__;
        PyObject * add(PyObject *, PyObject *);

        // build a group out of the difference of two others
        extern const char * const subtract__name__;
        extern const char * const subtract__doc__;
        PyObject * subtract(PyObject *, PyObject *);

        // build a group out of the intersection of two others
        extern const char * const intersect__name__;
        extern const char * const intersect__doc__;
        PyObject * intersect(PyObject *, PyObject *);

    } // of namespace group
} // of namespace mpi

#endif

// end of file
