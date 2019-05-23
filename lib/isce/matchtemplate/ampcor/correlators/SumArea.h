// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(ampcor_libampcor_correlators_sumarea_h)
#define ampcor_libampcor_correlators_sumarea_h


// externals
#include <complex>
#include <pyre/grid.h>

// encapsulation of SumArea tables
template <typename rasterT>
class ampcor::correlators::SumArea {
    // types
public:
    // my client raster type
    using raster_type = rasterT;

    // my pixel type
    using pixel_type = typename raster_type::cell_type;
    // my grid
    using grid_type = heapgrid_t<raster_type::dim(), pixel_type>;
    // my index type
    using index_type = typename grid_type::index_type;
    // my slice type
    using slice_type = typename grid_type::slice_type;

    // for sizing things
    using size_type = size_t; // from ampcor::dom

    // interface
public:
    // access to my layout
    const auto & layout() const;

    // the location of the data buffer
    inline auto data() const;

    // data access
    inline const auto & operator[](size_type offset) const;
    inline const auto & operator[](const index_type & index) const;

    inline auto & operator[](size_type offset);
    inline auto & operator[](const index_type & index);

    // compute the sum of the values within a window; note that, in our convention, the window
    // specification is given by [ slice.low(), slice.high() )
    inline auto sum(const slice_type & slice) const;

    // meta-methods
public:
    inline SumArea(const raster_type & raster);

    // implementation details: data
private:
    grid_type _grid;
};


// code guard
#endif

// end of file
