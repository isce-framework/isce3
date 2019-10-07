// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(gsl_extension_rng_h)
#define gsl_extension_rng_h


// place everything in my private namespace
namespace gsl {
    namespace rng {

        // initialization
        void initialize();

        // avail
        extern const char * const avail__name__;
        extern const char * const avail__doc__;
        PyObject * avail(PyObject *, PyObject *);

        // alloc
        extern const char * const alloc__name__;
        extern const char * const alloc__doc__;
        PyObject * alloc(PyObject *, PyObject *);

        // set
        extern const char * const set__name__;
        extern const char * const set__doc__;
        PyObject * set(PyObject *, PyObject *);

        // name
        extern const char * const name__name__;
        extern const char * const name__doc__;
        PyObject * name(PyObject *, PyObject *);

        // range
        extern const char * const range__name__;
        extern const char * const range__doc__;
        PyObject * range(PyObject *, PyObject *);

        // get
        extern const char * const get__name__;
        extern const char * const get__doc__;
        PyObject * get(PyObject *, PyObject *);

        // uniform
        extern const char * const uniform__name__;
        extern const char * const uniform__doc__;
        PyObject * uniform(PyObject *, PyObject *);

    } // of namespace rng
} // of namespace gsl

#endif

// end of file
