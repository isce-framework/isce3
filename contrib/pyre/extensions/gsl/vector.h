// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(gsl_extension_vector_h)
#define gsl_extension_vector_h


// place everything in my private namespace
namespace gsl {
    namespace vector {

        // alloc
        extern const char * const alloc__name__;
        extern const char * const alloc__doc__;
        PyObject * alloc(PyObject *, PyObject *);

        // view_alloc
        extern const char * const view_alloc__name__;
        extern const char * const view_alloc__doc__;
        PyObject * view_alloc(PyObject *, PyObject *);

        // set_zero
        extern const char * const zero__name__;
        extern const char * const zero__doc__;
        PyObject * zero(PyObject *, PyObject *);

        // set_all
        extern const char * const fill__name__;
        extern const char * const fill__doc__;
        PyObject * fill(PyObject *, PyObject *);

        // vector_basis
        extern const char * const basis__name__;
        extern const char * const basis__doc__;
        PyObject * basis(PyObject *, PyObject *);

        // vector_copy
        extern const char * const copy__name__;
        extern const char * const copy__doc__;
        PyObject * copy(PyObject *, PyObject *);

        // vector_tuple
        extern const char * const tuple__name__;
        extern const char * const tuple__doc__;
        PyObject * tuple(PyObject *, PyObject *);

        // vector_read
        extern const char * const read__name__;
        extern const char * const read__doc__;
        PyObject * read(PyObject *, PyObject *);

        // vector_write
        extern const char * const write__name__;
        extern const char * const write__doc__;
        PyObject * write(PyObject *, PyObject *);

        // vector_scanf
        extern const char * const scanf__name__;
        extern const char * const scanf__doc__;
        PyObject * scanf(PyObject *, PyObject *);

        // vector_print
        extern const char * const printf__name__;
        extern const char * const printf__doc__;
        PyObject * printf(PyObject *, PyObject *);

        // vector_get
        extern const char * const get__name__;
        extern const char * const get__doc__;
        PyObject * get(PyObject *, PyObject *);

        // vector_set
        extern const char * const set__name__;
        extern const char * const set__doc__;
        PyObject * set(PyObject *, PyObject *);

        // vector_contains
        extern const char * const contains__name__;
        extern const char * const contains__doc__;
        PyObject * contains(PyObject *, PyObject *);

        // vector_max
        extern const char * const max__name__;
        extern const char * const max__doc__;
        PyObject * max(PyObject *, PyObject *);

        // vector_min
        extern const char * const min__name__;
        extern const char * const min__doc__;
        PyObject * min(PyObject *, PyObject *);

        // vector_minmax
        extern const char * const minmax__name__;
        extern const char * const minmax__doc__;
        PyObject * minmax(PyObject *, PyObject *);

        // vector_equal
        extern const char * const equal__name__;
        extern const char * const equal__doc__;
        PyObject * equal(PyObject *, PyObject *);

        // vector_add
        extern const char * const add__name__;
        extern const char * const add__doc__;
        PyObject * add(PyObject *, PyObject *);

        // vector_sub
        extern const char * const sub__name__;
        extern const char * const sub__doc__;
        PyObject * sub(PyObject *, PyObject *);

        // vector_mul
        extern const char * const mul__name__;
        extern const char * const mul__doc__;
        PyObject * mul(PyObject *, PyObject *);

        // vector_div
        extern const char * const div__name__;
        extern const char * const div__doc__;
        PyObject * div(PyObject *, PyObject *);

        // vector_add_constant
        extern const char * const shift__name__;
        extern const char * const shift__doc__;
        PyObject * shift(PyObject *, PyObject *);

        // vector_scale
        extern const char * const scale__name__;
        extern const char * const scale__doc__;
        PyObject * scale(PyObject *, PyObject *);

        // vector_dataptr
        extern const char * const dataptr__name__;
        extern const char * const dataptr__doc__;
        PyObject * dataptr(PyObject *, PyObject *);

        // statistics
        // vector_sort
        extern const char * const sort__name__;
        extern const char * const sort__doc__;
        PyObject * sort(PyObject *, PyObject *);

        // vector_sortIndex
        extern const char * const sortIndex__name__;
        extern const char * const sortIndex__doc__;
        PyObject * sortIndex(PyObject *, PyObject *);

        // vector_mean
        extern const char * const mean__name__;
        extern const char * const mean__doc__;
        PyObject * mean(PyObject *, PyObject *);

        // vector_median
        extern const char * const median__name__;
        extern const char * const median__doc__;
        PyObject * median(PyObject *, PyObject *);

        // vector_variance
        extern const char * const variance__name__;
        extern const char * const variance__doc__;
        PyObject * variance(PyObject *, PyObject *);

        // vector_sdev
        extern const char * const sdev__name__;
        extern const char * const sdev__doc__;
        PyObject * sdev(PyObject *, PyObject *);

    } // of namespace vector
} // of namespace gsl

#endif

// end of file
