// -*- C++ -*-
//
// michael a.g. aïvázis @ orthologue
// Lijun Zhu @ Caltech
// (c) 1998-2019 all rights reserved
//

#if !defined(gsl_extension_stats_h)
#define gsl_extension_stats_h


// place everything in my private namespace
namespace gsl {
    namespace stats {

        // gsl_stats_correlation
        extern const char * const correlation__name__;
        extern const char * const correlation__doc__;
        PyObject * correlation(PyObject *, PyObject *);

        // gsl_stats_covariance
        extern const char * const covariance__name__;
        extern const char * const covariance__doc__;
        PyObject * covariance(PyObject *, PyObject *);

        // gsl_stats_matrix_mean
        extern const char * const matrix_mean__name__;
        extern const char * const matrix_mean__doc__;
        PyObject * matrix_mean(PyObject *, PyObject *);

        // gsl_stats_matrix_mean_sd
        extern const char * const matrix_mean_sd__name__;
        extern const char * const matrix_mean_sd__doc__;
        PyObject * matrix_mean_sd(PyObject *, PyObject *);

        // gsl_stats_matrix_mean_std
        extern const char * const matrix_mean_std__name__;
        extern const char * const matrix_mean_std__doc__;
        PyObject * matrix_mean_std(PyObject *, PyObject *);


    } // of namespace stats
} // of namespace gsl

#endif

// end of file
