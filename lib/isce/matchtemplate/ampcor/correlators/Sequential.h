// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(ampcor_libampcor_correlators_sequential_h)
#define ampcor_libampcor_correlators_sequential_h


// access to the dom
#include <isce/matchtemplate/ampcor/dom.h>

// resource management and orchestration of the execution of the correlation plan
class ampcor::correlators::Sequential {
    // types
public:
    // my storage type
    using cell_type = double;
    // my client raster type
    using slc_type = ampcor::dom::slc_t;
    // for describing slices of rasters
    using slice_type = slc_type::slice_type;
    // for describing the shapes of tiles
    using shape_type = slc_type::shape_type;
    // for index arithmetic
    using index_type = slc_type::index_type;
    // for sizing things
    using size_type = slc_type::size_type;

    // i use {cell_type} grids that ride on top of my dataspace with the same layout as the SLC
    using gview_type = pyre::grid::grid_t<cell_type,
                                          slc_type::layout_type,
                                          pyre::memory::view_t<cell_type>>;

    // for the sum area table, use a grid on the heap
    using grid_type = heapgrid_t<slc_type::layout_type::dim(), cell_type>;
    // sum area tables
    using sat_type = sumarea_t<grid_type>;

    // interface
public:
    // add a reference tile to the pile
    inline void addReferenceTile(const slc_type & slc, size_type pid, slice_type slice);
    // add a target search window to the pile
    inline void addTargetTile(const slc_type & slc, size_type pid, slice_type slice);

    // compute pixel level adjustments to the registration map
    void adjust();
    // compute sub-pixel level refinements to the registration map
    void refine();

    // meta-methods
public:
    virtual ~Sequential();
    Sequential(size_type pairs, const shape_type & refShape, const shape_type & tgtShape);

    // implementation details: data
private:
    // my capacity, in {ref/tgt} pairs
    size_type _pairs;

    // the shape of the reference tiles
    shape_type _refShape;
    // the shape of the search windows in the target image
    shape_type _tgtShape;

    // the number of cells in a reference tile
    size_type _refCells;
    // the number of cells in a target search window
    size_type _tgtCells;

    // storage for the tile pairs
    cell_type * _buffer;
    // storage for the correlation results
    cell_type * _correlation;
};


// code guard
#endif

// end of file
