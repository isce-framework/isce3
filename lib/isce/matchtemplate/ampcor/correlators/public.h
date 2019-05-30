// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(ampcor_libampcor_correlators_public_h)
#define ampcor_libampcor_correlators_public_h


// externals
// STL
#include <cmath>
#include <algorithm>
#include <functional>
#include <numeric>
// pyre
#include <pyre/journal.h>
#include <pyre/timers.h>
#include <pyre/grid.h>
// access to the dom
#include <isce/matchtemplate/ampcor/dom.h>

namespace ampcor {
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
        // correlator
        template <typename rasterT>
        class Correlator;
        // workers
        class Sequential;
        // sum area
        template <typename rasterT>
        class SumArea;

        // the public type aliases for the local objects
        // correlator
        template <typename rasterT>
        using correlator_t = Correlator<rasterT>;
        // workers
        using sequential_t = Sequential;
        // sum area
        template <typename rasterT>
        using sumarea_t = SumArea<rasterT>;

    } // of namespace correlators
} // of namespace ampcor


// the class declarations
#include "Correlator.h"
#include "Sequential.h"
#include "SumArea.h"

// the implementations of the inline methods
// correlators
#define ampcor_libampcor_correlators_correlator_icc
#include "Correlator.icc"
#undef ampcor_libampcor_correlators_correlator_icc

// workers
#define ampcor_libampcor_correlators_sequential_icc
#include "Sequential.icc"
#undef ampcor_libampcor_correlators_sequential_icc

// sum area tables
#define ampcor_libampcor_correlators_sumarea_icc
#include "SumArea.icc"
#undef ampcor_libampcor_correlators_sumarea_icc


// code guard
#endif

// end of file
