// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(gsl_extension_permutation_h)
#define gsl_extension_permutation_h


// place everything in my private namespace
namespace gsl {
    namespace permutation {

        // alloc
        extern const char * const alloc__name__;
        extern const char * const alloc__doc__;
        PyObject * alloc(PyObject *, PyObject *);

        // init
        extern const char * const init__name__;
        extern const char * const init__doc__;
        PyObject * init(PyObject *, PyObject *);

        // copy
        extern const char * const copy__name__;
        extern const char * const copy__doc__;
        PyObject * copy(PyObject *, PyObject *);

        // get
        extern const char * const get__name__;
        extern const char * const get__doc__;
        PyObject * get(PyObject *, PyObject *);

        // swap
        extern const char * const swap__name__;
        extern const char * const swap__doc__;
        PyObject * swap(PyObject *, PyObject *);

        // size
        extern const char * const size__name__;
        extern const char * const size__doc__;
        PyObject * size(PyObject *, PyObject *);

        // valid
        extern const char * const valid__name__;
        extern const char * const valid__doc__;
        PyObject * valid(PyObject *, PyObject *);

        // reverse
        extern const char * const reverse__name__;
        extern const char * const reverse__doc__;
        PyObject * reverse(PyObject *, PyObject *);

        // inverse
        extern const char * const inverse__name__;
        extern const char * const inverse__doc__;
        PyObject * inverse(PyObject *, PyObject *);

        // next
        extern const char * const next__name__;
        extern const char * const next__doc__;
        PyObject * next(PyObject *, PyObject *);

        // prev
        extern const char * const prev__name__;
        extern const char * const prev__doc__;
        PyObject * prev(PyObject *, PyObject *);

    } // of namespace permutation
} // of namespace gsl

#endif

// end of file
