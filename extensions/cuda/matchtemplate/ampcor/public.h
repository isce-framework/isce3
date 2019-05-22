// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(ampcor_libampcor_cuda_correlators_public_h)
#define ampcor_libampcor_cuda_correlators_public_h


// externals
// STL
#include <cmath>
#include <algorithm>
#include <functional>
#include <numeric>
#include <exception>
// cuda
#include <cuda_runtime.h>
// pyre
#include <pyre/journal.h>
#include <pyre/timers.h>
#include <pyre/grid.h>
// access to the dom
#include <isce/matchtemplate/ampcor/dom.h>

namespace ampcor {
    namespace cuda {
        namespace correlators {

            // local type aliases
            // sizes of things
            using size_t = std::size_t;

            // pyre timers
            using timer_t = pyre::timer_t;

            // a simple grid on the heap
            template <size_t dim, typename pixel_t>
            using heapgrid_t =
                pyre::grid::grid_t< pixel_t,
                                    pyre::grid::layout_t<
                                        pyre::grid::index_t<std::array<size_t, dim>>>,
                                    pyre::memory::heap_t<pixel_t>
                                    >;

            // forward declarations of local classes
            // the manager
            template <typename raster_t> class Sequential;

            // the public type aliases for the local objects
            // workers
            template <typename raster_t>
            using sequential_t = Sequential<raster_t>;

        } // of namespace correlators
    } // of namespace cuda
} // of namespace ampcor

// kernels
#include "kernels.h"

// the class declarations
#include "Sequential.h"

// the inline definitions
// sequential
#define ampcor_cuda_correlators_Sequential_icc
#include "Sequential.icc"
#undef ampcor_cuda_correlators_Sequential_icc


// code guard
#endif

// end of file
