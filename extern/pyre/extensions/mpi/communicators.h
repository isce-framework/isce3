// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//


#if !defined(pyre_extensions_mpi_communicators_h_)
#define pyre_extensions_mpi_communicators_h_


// place everything in my private namespace
namespace mpi {
    namespace communicator {

        // the predefined communicators
        extern PyObject * world;

        // create a communicator (MPI_Comm_create)
        extern const char * const create__name__;
        extern const char * const create__doc__;
        PyObject * create(PyObject *, PyObject *);

        // return the communicator size (MPI_Comm_size)
        extern const char * const size__name__;
        extern const char * const size__doc__;
        PyObject * size(PyObject *, PyObject *);

        // return the process rank in a given communicator (MPI_Comm_rank)
        extern const char * const rank__name__;
        extern const char * const rank__doc__;
        PyObject * rank(PyObject *, PyObject *);

        // set a communicator barrier (MPI_Barrier)
        extern const char * const barrier__name__;
        extern const char * const barrier__doc__;
        PyObject * barrier(PyObject *, PyObject *);

        // broadcast a python object to all tasks
        extern const char * const bcast__name__;
        extern const char * const bcast__doc__;
        PyObject * bcast(PyObject *, PyObject *);

        // sum reduction
        extern const char * const sum__name__;
        extern const char * const sum__doc__;
        PyObject * sum(PyObject *, PyObject *);

        // product reduction
        extern const char * const product__name__;
        extern const char * const product__doc__;
        PyObject * product(PyObject *, PyObject *);

        // max reduction
        extern const char * const max__name__;
        extern const char * const max__doc__;
        PyObject * max(PyObject *, PyObject *);

        // min reduction
        extern const char * const min__name__;
        extern const char * const min__doc__;
        PyObject * min(PyObject *, PyObject *);

        // sum reduction and distribute to all
        extern const char * const sum_all__name__;
        extern const char * const sum_all__doc__;
        PyObject * sum_all(PyObject *, PyObject *);

        // product reduction and distribute to all
        extern const char * const product_all__name__;
        extern const char * const product_all__doc__;
        PyObject * product_all(PyObject *, PyObject *);

        // max reduction and distribute to all
        extern const char * const max_all__name__;
        extern const char * const max_all__doc__;
        PyObject * max_all(PyObject *, PyObject *);

        // min reduction and distribute to all
        extern const char * const min_all__name__;
        extern const char * const min_all__doc__;
        PyObject * min_all(PyObject *, PyObject *);

    } // of namespace communicator

    namespace cartesian {
        // create a cartesian communicator (MPI_Cart_create)
        extern const char * const create__name__;
        extern const char * const create__doc__;
        PyObject * create(PyObject *, PyObject *);

        // return the coordinates of the process in the cartesian communicator (MPI_Cart_coords)
        extern const char * const coordinates__name__;
        extern const char * const coordinates__doc__;
        PyObject * coordinates(PyObject *, PyObject *);
    } // of namespace cartesian

} // of namespace mpi

#endif

// end of file
