// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(gsl_extension_blas_h)
#define gsl_extension_blas_h


// place everything in my private namespace
namespace gsl {
    namespace blas {

        // level 1
        extern const char * const ddot__name__;
        extern const char * const ddot__doc__;
        PyObject * ddot(PyObject *, PyObject *);

        extern const char * const dnrm2__name__;
        extern const char * const dnrm2__doc__;
        PyObject * dnrm2(PyObject *, PyObject *);

        extern const char * const dasum__name__;
        extern const char * const dasum__doc__;
        PyObject * dasum(PyObject *, PyObject *);

        extern const char * const idamax__name__;
        extern const char * const idamax__doc__;
        PyObject * idamax(PyObject *, PyObject *);

        extern const char * const dswap__name__;
        extern const char * const dswap__doc__;
        PyObject * dswap(PyObject *, PyObject *);

        extern const char * const dcopy__name__;
        extern const char * const dcopy__doc__;
        PyObject * dcopy(PyObject *, PyObject *);

        extern const char * const daxpy__name__;
        extern const char * const daxpy__doc__;
        PyObject * daxpy(PyObject *, PyObject *);

        extern const char * const dscal__name__;
        extern const char * const dscal__doc__;
        PyObject * dscal(PyObject *, PyObject *);

        extern const char * const drotg__name__;
        extern const char * const drotg__doc__;
        PyObject * drotg(PyObject *, PyObject *);

        extern const char * const drot__name__;
        extern const char * const drot__doc__;
        PyObject * drot(PyObject *, PyObject *);

        // level 2
        extern const char * const dgemv__name__;
        extern const char * const dgemv__doc__;
        PyObject * dgemv(PyObject *, PyObject *);

        extern const char * const dtrmv__name__;
        extern const char * const dtrmv__doc__;
        PyObject * dtrmv(PyObject *, PyObject *);

        extern const char * const dtrsv__name__;
        extern const char * const dtrsv__doc__;
        PyObject * dtrsv(PyObject *, PyObject *);

        extern const char * const dsymv__name__;
        extern const char * const dsymv__doc__;
        PyObject * dsymv(PyObject *, PyObject *);

        extern const char * const dsyr__name__;
        extern const char * const dsyr__doc__;
        PyObject * dsyr(PyObject *, PyObject *);

        // level 3
        extern const char * const dgemm__name__;
        extern const char * const dgemm__doc__;
        PyObject * dgemm(PyObject *, PyObject *);
        
        extern const char * const dsymm__name__;
        extern const char * const dsymm__doc__;
        PyObject * dsymm(PyObject *, PyObject *);

        extern const char * const dtrmm__name__;
        extern const char * const dtrmm__doc__;
        PyObject * dtrmm(PyObject *, PyObject *);

    } // of namespace blas
} // of namespace gsl

#endif

// end of file
