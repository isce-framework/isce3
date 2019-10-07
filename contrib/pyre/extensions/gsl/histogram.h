// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(gsl_extension_histogram_h)
#define gsl_extension_histogram_h


// place everything in my private namespace
namespace gsl {
    namespace histogram {

        // alloc
        extern const char * const alloc__name__;
        extern const char * const alloc__doc__;
        PyObject * alloc(PyObject *, PyObject *);

        // uniform
        extern const char * const uniform__name__;
        extern const char * const uniform__doc__;
        PyObject * uniform(PyObject *, PyObject *);

        // ranges
        extern const char * const ranges__name__;
        extern const char * const ranges__doc__;
        PyObject * ranges(PyObject *, PyObject *);

        // reset
        extern const char * const reset__name__;
        extern const char * const reset__doc__;
        PyObject * reset(PyObject *, PyObject *);

        // increment
        extern const char * const increment__name__;
        extern const char * const increment__doc__;
        PyObject * increment(PyObject *, PyObject *);

        // accumulate
        extern const char * const accumulate__name__;
        extern const char * const accumulate__doc__;
        PyObject * accumulate(PyObject *, PyObject *);

        // fill
        extern const char * const fill__name__;
        extern const char * const fill__doc__;
        PyObject * fill(PyObject *, PyObject *);

        // clone
        extern const char * const clone__name__;
        extern const char * const clone__doc__;
        PyObject * clone(PyObject *, PyObject *);

        // copy
        extern const char * const copy__name__;
        extern const char * const copy__doc__;
        PyObject * copy(PyObject *, PyObject *);

        // vector
        extern const char * const vector__name__;
        extern const char * const vector__doc__;
        PyObject * vector(PyObject *, PyObject *);

        // find
        extern const char * const find__name__;
        extern const char * const find__doc__;
        PyObject * find(PyObject *, PyObject *);

        // max
        extern const char * const max__name__;
        extern const char * const max__doc__;
        PyObject * max(PyObject *, PyObject *);

        // min
        extern const char * const min__name__;
        extern const char * const min__doc__;
        PyObject * min(PyObject *, PyObject *);

        // range
        extern const char * const range__name__;
        extern const char * const range__doc__;
        PyObject * range(PyObject *, PyObject *);

        // max_bin
        extern const char * const max_bin__name__;
        extern const char * const max_bin__doc__;
        PyObject * max_bin(PyObject *, PyObject *);

        // min_bin
        extern const char * const min_bin__name__;
        extern const char * const min_bin__doc__;
        PyObject * min_bin(PyObject *, PyObject *);

        // max_val
        extern const char * const max_val__name__;
        extern const char * const max_val__doc__;
        PyObject * max_val(PyObject *, PyObject *);

        // min_val
        extern const char * const min_val__name__;
        extern const char * const min_val__doc__;
        PyObject * min_val(PyObject *, PyObject *);

        // mean
        extern const char * const mean__name__;
        extern const char * const mean__doc__;
        PyObject * mean(PyObject *, PyObject *);

        // sdev
        extern const char * const sdev__name__;
        extern const char * const sdev__doc__;
        PyObject * sdev(PyObject *, PyObject *);

        // sum
        extern const char * const sum__name__;
        extern const char * const sum__doc__;
        PyObject * sum(PyObject *, PyObject *);

        // get
        extern const char * const get__name__;
        extern const char * const get__doc__;
        PyObject * get(PyObject *, PyObject *);

        // add
        extern const char * const add__name__;
        extern const char * const add__doc__;
        PyObject * add(PyObject *, PyObject *);

        // sub
        extern const char * const sub__name__;
        extern const char * const sub__doc__;
        PyObject * sub(PyObject *, PyObject *);

        // mul
        extern const char * const mul__name__;
        extern const char * const mul__doc__;
        PyObject * mul(PyObject *, PyObject *);

        // div
        extern const char * const div__name__;
        extern const char * const div__doc__;
        PyObject * div(PyObject *, PyObject *);

        // shift
        extern const char * const shift__name__;
        extern const char * const shift__doc__;
        PyObject * shift(PyObject *, PyObject *);

        // scale
        extern const char * const scale__name__;
        extern const char * const scale__doc__;
        PyObject * scale(PyObject *, PyObject *);

    } // of namespace histogram
} // of namespace gsl

#endif

// end of file
