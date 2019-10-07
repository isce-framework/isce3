// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// for the build system
#include <portinfo>
// external dependencies
#include <iostream>
#include <string>
#include <Python.h>
#include <gsl/gsl_errno.h>

// the module method declarations
#include "exceptions.h"
#include "metadata.h"

#include "blas.h" // blas
#include "histogram.h" // linear algebra
#include "linalg.h" // linear algebra
#include "matrix.h" // matrices
#include "pdf.h" // probability distribution functions
#include "permutation.h" // permutations
#include "rng.h" // random numbers
#include "vector.h" // vectors
#include "stats.h" // stats
#include "numpy.h" // numpy hook

// mpi support
#if defined(WITH_MPI)
#include "partition.h"
#endif

// put everything in my private namespace
namespace gsl {

    // the module method table
    PyMethodDef module_methods[] = {
        // module metadata
        // the copyright method
        { copyright__name__, copyright, METH_VARARGS, copyright__doc__ },
        // the license
        { license__name__, license, METH_VARARGS, license__doc__ },
        // the version
        { version__name__, version, METH_VARARGS, version__doc__ },

        // blas - level 1
        { blas::ddot__name__, blas::ddot, METH_VARARGS, blas::ddot__doc__ },
        { blas::dnrm2__name__, blas::dnrm2, METH_VARARGS, blas::dnrm2__doc__ },
        { blas::dasum__name__, blas::dasum, METH_VARARGS, blas::dasum__doc__ },
        { blas::idamax__name__, blas::idamax, METH_VARARGS, blas::idamax__doc__ },
        { blas::dswap__name__, blas::dswap, METH_VARARGS, blas::dswap__doc__ },
        { blas::dcopy__name__, blas::dcopy, METH_VARARGS, blas::dcopy__doc__ },
        { blas::daxpy__name__, blas::daxpy, METH_VARARGS, blas::daxpy__doc__ },
        { blas::dscal__name__, blas::dscal, METH_VARARGS, blas::dscal__doc__ },
        { blas::drotg__name__, blas::drotg, METH_VARARGS, blas::drotg__doc__ },
        { blas::drot__name__, blas::drot, METH_VARARGS, blas::drot__doc__ },
        // blas - level 2
        { blas::dgemv__name__, blas::dgemv, METH_VARARGS, blas::dgemv__doc__ },
        { blas::dtrmv__name__, blas::dtrmv, METH_VARARGS, blas::dtrmv__doc__ },
        { blas::dtrsv__name__, blas::dtrsv, METH_VARARGS, blas::dtrsv__doc__ },
        { blas::dsymv__name__, blas::dsymv, METH_VARARGS, blas::dsymv__doc__ },
        { blas::dsyr__name__, blas::dsyr, METH_VARARGS, blas::dsyr__doc__ },
        // blas - level 3
        { blas::dgemm__name__, blas::dgemm, METH_VARARGS, blas::dgemm__doc__ },
        { blas::dsymm__name__, blas::dsymm, METH_VARARGS, blas::dsymm__doc__ },
        { blas::dtrmm__name__, blas::dtrmm, METH_VARARGS, blas::dtrmm__doc__ },

        // histograms
        { histogram::alloc__name__, histogram::alloc, METH_VARARGS, histogram::alloc__doc__ },
        { histogram::uniform__name__, histogram::uniform, METH_VARARGS, histogram::uniform__doc__ },
        { histogram::ranges__name__, histogram::ranges, METH_VARARGS, histogram::ranges__doc__ },
        { histogram::reset__name__, histogram::reset, METH_VARARGS, histogram::reset__doc__ },
        { histogram::increment__name__, histogram::increment, METH_VARARGS,
          histogram::increment__doc__ },
        { histogram::accumulate__name__, histogram::accumulate, METH_VARARGS,
          histogram::accumulate__doc__ },
        { histogram::fill__name__, histogram::fill, METH_VARARGS, histogram::fill__doc__ },
        { histogram::clone__name__, histogram::clone, METH_VARARGS, histogram::clone__doc__ },
        { histogram::copy__name__, histogram::copy, METH_VARARGS, histogram::copy__doc__ },
        { histogram::vector__name__, histogram::vector, METH_VARARGS, histogram::vector__doc__ },
        { histogram::find__name__, histogram::find, METH_VARARGS, histogram::find__doc__ },
        { histogram::max__name__, histogram::max, METH_VARARGS, histogram::max__doc__ },
        { histogram::min__name__, histogram::min, METH_VARARGS, histogram::min__doc__ },
        { histogram::range__name__, histogram::range, METH_VARARGS, histogram::range__doc__ },
        { histogram::max_bin__name__, histogram::max_bin, METH_VARARGS, histogram::max_bin__doc__ },
        { histogram::min_bin__name__, histogram::min_bin, METH_VARARGS, histogram::min_bin__doc__ },
        { histogram::max_val__name__, histogram::max_val, METH_VARARGS, histogram::max_val__doc__ },
        { histogram::min_val__name__, histogram::min_val, METH_VARARGS, histogram::min_val__doc__ },
        { histogram::mean__name__, histogram::mean, METH_VARARGS, histogram::mean__doc__ },
        { histogram::sdev__name__, histogram::sdev, METH_VARARGS, histogram::sdev__doc__ },
        { histogram::sum__name__, histogram::sum, METH_VARARGS, histogram::sum__doc__ },
        { histogram::get__name__, histogram::get, METH_VARARGS, histogram::get__doc__ },
        { histogram::add__name__, histogram::add, METH_VARARGS, histogram::add__doc__ },
        { histogram::sub__name__, histogram::sub, METH_VARARGS, histogram::sub__doc__ },
        { histogram::mul__name__, histogram::mul, METH_VARARGS, histogram::mul__doc__ },
        { histogram::div__name__, histogram::div, METH_VARARGS, histogram::div__doc__ },
        { histogram::shift__name__, histogram::shift, METH_VARARGS, histogram::shift__doc__ },
        { histogram::scale__name__, histogram::scale, METH_VARARGS, histogram::scale__doc__ },

        // linalg
        { linalg::LU_decomp__name__, linalg::LU_decomp, METH_VARARGS, linalg::LU_decomp__doc__ },
        { linalg::LU_invert__name__, linalg::LU_invert, METH_VARARGS, linalg::LU_invert__doc__ },
        { linalg::LU_det__name__, linalg::LU_det, METH_VARARGS, linalg::LU_det__doc__ },
        { linalg::LU_lndet__name__, linalg::LU_lndet, METH_VARARGS, linalg::LU_lndet__doc__ },
        { linalg::cholesky_decomp__name__, linalg::cholesky_decomp, METH_VARARGS,
          linalg::cholesky_decomp__doc__ },

        // matrices
        { matrix::alloc__name__, matrix::alloc, METH_VARARGS, matrix::alloc__doc__ },
        { matrix::view_alloc__name__, matrix::view_alloc, METH_VARARGS, matrix::view_alloc__doc__ },
        { matrix::zero__name__, matrix::zero, METH_VARARGS, matrix::zero__doc__ },
        { matrix::fill__name__, matrix::fill, METH_VARARGS, matrix::fill__doc__ },
        { matrix::identity__name__, matrix::identity, METH_VARARGS, matrix::identity__doc__ },
        { matrix::copy__name__, matrix::copy, METH_VARARGS, matrix::copy__doc__ },
        { matrix::tuple__name__, matrix::tuple, METH_VARARGS, matrix::tuple__doc__ },
        { matrix::read__name__, matrix::read, METH_VARARGS, matrix::read__doc__ },
        { matrix::write__name__, matrix::write, METH_VARARGS, matrix::write__doc__ },
        { matrix::scanf__name__, matrix::scanf, METH_VARARGS, matrix::scanf__doc__ },
        { matrix::printf__name__, matrix::printf, METH_VARARGS, matrix::printf__doc__ },
        { matrix::transpose__name__, matrix::transpose, METH_VARARGS, matrix::transpose__doc__ },
        { matrix::get__name__, matrix::get, METH_VARARGS, matrix::get__doc__ },
        { matrix::set__name__, matrix::set, METH_VARARGS, matrix::set__doc__ },
        { matrix::get_col__name__, matrix::get_col, METH_VARARGS, matrix::get_col__doc__ },
        { matrix::get_row__name__, matrix::get_row, METH_VARARGS, matrix::get_row__doc__ },
        { matrix::set_col__name__, matrix::set_col, METH_VARARGS, matrix::set_col__doc__ },
        { matrix::set_row__name__, matrix::set_row, METH_VARARGS, matrix::set_row__doc__ },
        { matrix::contains__name__, matrix::contains, METH_VARARGS, matrix::contains__doc__ },
        { matrix::max__name__, matrix::max, METH_VARARGS, matrix::max__doc__ },
        { matrix::min__name__, matrix::min, METH_VARARGS, matrix::min__doc__ },
        { matrix::minmax__name__, matrix::minmax, METH_VARARGS, matrix::minmax__doc__ },
        { matrix::equal__name__, matrix::equal, METH_VARARGS, matrix::equal__doc__ },
        { matrix::add__name__, matrix::add, METH_VARARGS, matrix::add__doc__ },
        { matrix::sub__name__, matrix::sub, METH_VARARGS, matrix::sub__doc__ },
        { matrix::mul__name__, matrix::mul, METH_VARARGS, matrix::mul__doc__ },
        { matrix::div__name__, matrix::div, METH_VARARGS, matrix::div__doc__ },
        { matrix::shift__name__, matrix::shift, METH_VARARGS, matrix::shift__doc__ },
        { matrix::scale__name__, matrix::scale, METH_VARARGS, matrix::scale__doc__ },
        { matrix::eigen_symmetric__name__, matrix::eigen_symmetric, METH_VARARGS,
          matrix::eigen_symmetric__doc__ },

        // permutations
        { permutation::alloc__name__, permutation::alloc, METH_VARARGS, permutation::alloc__doc__ },
        { permutation::init__name__, permutation::init, METH_VARARGS, permutation::init__doc__ },
        { permutation::copy__name__, permutation::copy, METH_VARARGS, permutation::copy__doc__ },
        { permutation::get__name__, permutation::get, METH_VARARGS, permutation::get__doc__ },
        { permutation::swap__name__, permutation::swap, METH_VARARGS, permutation::swap__doc__ },
        { permutation::size__name__, permutation::size, METH_VARARGS, permutation::size__doc__ },
        { permutation::valid__name__, permutation::valid, METH_VARARGS, permutation::valid__doc__ },
        { permutation::reverse__name__, permutation::reverse, METH_VARARGS,
          permutation::reverse__doc__ },
        { permutation::inverse__name__, permutation::inverse, METH_VARARGS,
          permutation::inverse__doc__ },
        { permutation::next__name__, permutation::next, METH_VARARGS, permutation::next__doc__ },
        { permutation::prev__name__, permutation::prev, METH_VARARGS, permutation::prev__doc__ },

        // probability distribution functions
        { pdf::uniform::sample__name__, pdf::uniform::sample, METH_VARARGS,
          pdf::uniform::sample__doc__ },
        { pdf::uniform::density__name__, pdf::uniform::density, METH_VARARGS,
          pdf::uniform::density__doc__ },
        { pdf::uniform::vector__name__, pdf::uniform::vector, METH_VARARGS,
          pdf::uniform::vector__doc__ },
        { pdf::uniform::matrix__name__, pdf::uniform::matrix, METH_VARARGS,
          pdf::uniform::matrix__doc__ },
        { pdf::uniform_pos::sample__name__, pdf::uniform_pos::sample, METH_VARARGS,
          pdf::uniform_pos::sample__doc__ },
        { pdf::uniform_pos::vector__name__, pdf::uniform_pos::vector, METH_VARARGS,
          pdf::uniform_pos::vector__doc__ },
        { pdf::uniform_pos::matrix__name__, pdf::uniform_pos::matrix, METH_VARARGS,
          pdf::uniform_pos::matrix__doc__ },
        { pdf::gaussian::sample__name__, pdf::gaussian::sample, METH_VARARGS,
          pdf::gaussian::sample__doc__ },
        { pdf::gaussian::density__name__, pdf::gaussian::density, METH_VARARGS,
          pdf::gaussian::density__doc__ },
        { pdf::gaussian::vector__name__, pdf::gaussian::vector, METH_VARARGS,
          pdf::gaussian::vector__doc__ },
        { pdf::gaussian::matrix__name__, pdf::gaussian::matrix, METH_VARARGS,
          pdf::gaussian::matrix__doc__ },
        { pdf::ugaussian::sample__name__, pdf::ugaussian::sample, METH_VARARGS,
          pdf::ugaussian::sample__doc__ },
        { pdf::ugaussian::density__name__, pdf::ugaussian::density, METH_VARARGS,
          pdf::ugaussian::density__doc__ },
        { pdf::ugaussian::vector__name__, pdf::ugaussian::vector, METH_VARARGS,
          pdf::ugaussian::vector__doc__ },
        { pdf::ugaussian::matrix__name__, pdf::ugaussian::matrix, METH_VARARGS,
          pdf::ugaussian::matrix__doc__ },
        { pdf::dirichlet::sample__name__, pdf::dirichlet::sample, METH_VARARGS,
          pdf::dirichlet::sample__doc__ },
        { pdf::dirichlet::density__name__, pdf::dirichlet::density, METH_VARARGS,
          pdf::dirichlet::density__doc__ },
        { pdf::dirichlet::vector__name__, pdf::dirichlet::vector, METH_VARARGS,
          pdf::dirichlet::vector__doc__ },
        { pdf::dirichlet::matrix__name__, pdf::dirichlet::matrix, METH_VARARGS,
          pdf::dirichlet::matrix__doc__ },

        // random numbers
        { rng::avail__name__, rng::avail, METH_VARARGS, rng::avail__doc__ },
        { rng::alloc__name__, rng::alloc, METH_VARARGS, rng::alloc__doc__ },
        { rng::get__name__, rng::get, METH_VARARGS, rng::get__doc__ },
        { rng::name__name__, rng::name, METH_VARARGS, rng::name__doc__ },
        { rng::range__name__, rng::range, METH_VARARGS, rng::range__doc__ },
        { rng::set__name__, rng::set, METH_VARARGS, rng::set__doc__ },
        { rng::uniform__name__, rng::uniform, METH_VARARGS, rng::uniform__doc__ },

        // vectors
        { vector::alloc__name__, vector::alloc, METH_VARARGS, vector::alloc__doc__ },
        { vector::view_alloc__name__, vector::view_alloc, METH_VARARGS, vector::view_alloc__doc__ },
        { vector::zero__name__, vector::zero, METH_VARARGS, vector::zero__doc__ },
        { vector::fill__name__, vector::fill, METH_VARARGS, vector::fill__doc__ },
        { vector::basis__name__, vector::basis, METH_VARARGS, vector::basis__doc__ },
        { vector::copy__name__, vector::copy, METH_VARARGS, vector::copy__doc__ },
        { vector::tuple__name__, vector::tuple, METH_VARARGS, vector::tuple__doc__ },
        { vector::read__name__, vector::read, METH_VARARGS, vector::read__doc__ },
        { vector::write__name__, vector::write, METH_VARARGS, vector::write__doc__ },
        { vector::scanf__name__, vector::scanf, METH_VARARGS, vector::scanf__doc__ },
        { vector::printf__name__, vector::printf, METH_VARARGS, vector::printf__doc__ },
        { vector::get__name__, vector::get, METH_VARARGS, vector::get__doc__ },
        { vector::set__name__, vector::set, METH_VARARGS, vector::set__doc__ },
        { vector::contains__name__, vector::contains, METH_VARARGS, vector::contains__doc__ },
        { vector::max__name__, vector::max, METH_VARARGS, vector::max__doc__ },
        { vector::min__name__, vector::min, METH_VARARGS, vector::min__doc__ },
        { vector::minmax__name__, vector::minmax, METH_VARARGS, vector::minmax__doc__ },
        { vector::equal__name__, vector::equal, METH_VARARGS, vector::equal__doc__ },
        { vector::add__name__, vector::add, METH_VARARGS, vector::add__doc__ },
        { vector::sub__name__, vector::sub, METH_VARARGS, vector::sub__doc__ },
        { vector::mul__name__, vector::mul, METH_VARARGS, vector::mul__doc__ },
        { vector::div__name__, vector::div, METH_VARARGS, vector::div__doc__ },
        { vector::shift__name__, vector::shift, METH_VARARGS, vector::shift__doc__ },
        { vector::scale__name__, vector::scale, METH_VARARGS, vector::scale__doc__ },
        // statistics
        { vector::sort__name__, vector::sort, METH_VARARGS, vector::sort__doc__ },
        { vector::sortIndex__name__, vector::sortIndex, METH_VARARGS, vector::sortIndex__doc__ },
        { vector::mean__name__, vector::mean, METH_VARARGS, vector::mean__doc__ },
        { vector::median__name__, vector::median, METH_VARARGS, vector::median__doc__ },
        { vector::variance__name__, vector::variance, METH_VARARGS, vector::variance__doc__ },
        { vector::sdev__name__, vector::sdev, METH_VARARGS, vector::sdev__doc__ },

        // numpy
        { vector::ndarray__name__, vector::ndarray, METH_VARARGS, vector::ndarray__doc__ },
        { matrix::ndarray__name__, matrix::ndarray, METH_VARARGS, matrix::ndarray__doc__ },

        // more statistics
        { stats::correlation__name__, stats::correlation, METH_VARARGS, stats::correlation__doc__},
        { stats::covariance__name__, stats::covariance, METH_VARARGS, stats::covariance__doc__},
        { stats::matrix_mean__name__, stats::matrix_mean, METH_VARARGS, stats::matrix_mean__doc__},
        { stats::matrix_mean_sd__name__, stats::matrix_mean_sd, METH_VARARGS, stats::matrix_mean_sd__doc__},
        { stats::matrix_mean_std__name__, stats::matrix_mean_std, METH_VARARGS, stats::matrix_mean_std__doc__},

        // mpi support
#if defined(WITH_MPI)
        // matrix partitioning
        { mpi::bcastMatrix__name__, mpi::bcastMatrix, METH_VARARGS, mpi::bcastMatrix__doc__ },
        { mpi::gatherMatrix__name__, mpi::gatherMatrix, METH_VARARGS, mpi::gatherMatrix__doc__ },
        { mpi::scatterMatrix__name__, mpi::scatterMatrix, METH_VARARGS, mpi::scatterMatrix__doc__ },
        // vector partitioning
        { mpi::bcastVector__name__, mpi::bcastVector, METH_VARARGS, mpi::bcastVector__doc__ },
        { mpi::gatherVector__name__, mpi::gatherVector, METH_VARARGS, mpi::gatherVector__doc__ },
        { mpi::scatterVector__name__, mpi::scatterVector, METH_VARARGS, mpi::scatterVector__doc__ },
#endif

        // sentinel
        {0, 0, 0, 0}
    };

    // the module documentation string
    const char * const __doc__ = "sample module documentation string";

    // the module definition structure
    PyModuleDef module_definition = {
        // header
        PyModuleDef_HEAD_INIT,
        // the name of the module
        "gsl",
        // the module documentation string
        __doc__,
        // size of the per-interpreter state of the module; -1 if this state is global
        -1,
        // the methods defined in this module
        module_methods
    };
} // of namespace gsl


// my error handler
static void errorHandler(const char * reason, const char * file, int line, int gsl_errno) {
    // for now, just print the reason for the error
    std::cerr << " ** ** ** GSL error: " << reason << std::endl;
    // throw an exception
    return;
}


// initialization function for the module
// *must* be called PyInit_gsl
PyMODINIT_FUNC
PyInit_gsl()
{
    // create the module
    PyObject * module = PyModule_Create(&gsl::module_definition);
    // check whether module creation succeeded and raise an exception if not
    if (!module) {
        return 0;
    }

    // otherwise, we have an initialized module
    // set the error handler
    gsl_set_error_handler(&errorHandler);
    // initialize the table of known random number generators
    gsl::rng::initialize();

    // return the newly created module
    return module;
}

// end of file
