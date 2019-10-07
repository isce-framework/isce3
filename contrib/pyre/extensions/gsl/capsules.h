// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(gsl_extension_capsules_h)
#define gsl_extension_capsules_h

// capsules
namespace gsl {

    // histogram
    namespace histogram {
        const char * const capsule_t = "gsl.histogram";
        void free(PyObject *);
    }
    // matrix
    namespace matrix {
        const char * const capsule_t = "gsl.matrix";
        const char * const view_t = "gsl.matrix.view";
        void free(PyObject *);
        void freeview(PyObject *);
    }
    // rng
    namespace rng {
        const char * const capsule_t = "gsl.rng";
        void free(PyObject *);
    }
    // permutations
    namespace permutation {
        const char * const capsule_t = "gsl.permutation";
        void free(PyObject *);
    }
    // vectors
    namespace vector {
        const char * const capsule_t = "gsl.vector";
        const char * const view_t = "gsl.vector.view";
        void free(PyObject *);
        void freeview(PyObject *);
    }
}
// local

#endif

// end of file
