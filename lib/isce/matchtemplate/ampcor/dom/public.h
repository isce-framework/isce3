// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(ampcor_libampcor_dom_public_h)
#define ampcor_libampcor_dom_public_h


// externals
#include <pyre/grid.h>

// forward declarations
namespace ampcor {
    namespace dom {

        // local type aliases
        // sizes of things
        using size_t = std::size_t;
        // filenames
        using uri_t = pyre::memory::uri_t;

        // a memory mapped grid
        template <size_t dim, typename pixel_t = double>
        using mmap_t =
            pyre::grid::directgrid_t< pixel_t,
                                      pyre::grid::layout_t<
                                          pyre::grid::index_t<std::array<size_t,dim>>>
                                      >;

        // a const memory mapped grid
        template <size_t dim, typename pixel_t = double>
        using constmmap_t =
            pyre::grid::directgrid_t< pixel_t,
                                      pyre::grid::layout_t<
                                          pyre::grid::index_t<std::array<size_t,dim>>>,
                                      pyre::memory::constdirect_t<pixel_t>
                                      >;

        // forward declarations of local classes
        // raster images
        class Raster;
        // SLC images
        class SLC;

        // the public type aliases for the local objects
        using slc_t = SLC;

    } // of namespace dom
} // of namespace ampcor


// the class declarations
#include "Raster.h"
#include "SLC.h"

// the implementations of the inline methods
// raster
#define ampcor_libampcor_dom_raster_icc
#include "Raster.icc"
#undef ampcor_libampcor_dom_raster_icc
// slc
#define ampcor_libampcor_dom_slc_icc
#include "SLC.icc"
#undef ampcor_libampcor_dom_slc_icc


// code guard
#endif

// end of file
