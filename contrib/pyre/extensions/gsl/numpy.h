// -*- C++ -*-
//
// Lijun Zhu (ljzhu@gps.caltech.edu)
//
// (c) 1998-2019 all rights reserved
//

#if !defined(gsl_extension_numpy_h)
#define gsl_extension_numpy_h


// place everything in my private namespace
namespace gsl {
    namespace vector {
        // vector_asnumpy
        extern const char * const ndarray__name__;
        extern const char * const ndarray__doc__;
        PyObject * ndarray(PyObject *, PyObject *);
    } // of namespace vector

    namespace matrix {
        // matrix_ndarray
        extern const char * const ndarray__name__;
        extern const char * const ndarray__doc__;
        PyObject * ndarray(PyObject *, PyObject *);
    } // of namespace matrix
} // of namespace gsl

#endif //gsl_extension_numpy_h

// end of file
