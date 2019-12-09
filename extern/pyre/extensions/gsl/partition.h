// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(gsl_extension_partition_h)
#define gsl_extension_partition_h


// place everything in my private namespace
namespace gsl {
    namespace mpi {

        // matrix bcast
        extern const char * const bcastMatrix__name__;
        extern const char * const bcastMatrix__doc__;
        PyObject * bcastMatrix(PyObject *, PyObject *);

        // matrix gather
        extern const char * const gatherMatrix__name__;
        extern const char * const gatherMatrix__doc__;
        PyObject * gatherMatrix(PyObject *, PyObject *);

        // matrix scatter
        extern const char * const scatterMatrix__name__;
        extern const char * const scatterMatrix__doc__;
        PyObject * scatterMatrix(PyObject *, PyObject *);

        // vector bcast
        extern const char * const bcastVector__name__;
        extern const char * const bcastVector__doc__;
        PyObject * bcastVector(PyObject *, PyObject *);

        // vector gather
        extern const char * const gatherVector__name__;
        extern const char * const gatherVector__doc__;
        PyObject * gatherVector(PyObject *, PyObject *);

        // vector scatter
        extern const char * const scatterVector__name__;
        extern const char * const scatterVector__doc__;
        PyObject * scatterVector(PyObject *, PyObject *);

    } // of namespace mpi
} // of namespace gsl

#endif

// end of file
