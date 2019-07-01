// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(ampcor_libampcor_cuda_correlators_Sequential_h)
#define ampcor_libampcor_cuda_correlators_Sequential_h


// resource management and orchestration of the execution of the correlation plan
template <typename raster_t>
class ampcor::cuda::correlators::Sequential {
    // types
public:
    // my client raster type
    using raster_type = raster_t;
    // views over it
    using view_type = typename raster_type::view_type;
    using constview_type = typename raster_type::constview_type;
    // the underlying pixel complex type
    using cell_type = typename raster_type::cell_type;
    // the support of the pixel complex type
    using value_type = typename cell_type::value_type;
    // for describing slices of rasters
    using slice_type = typename raster_type::slice_type;
    // for describing the shapes of tiles
    using shape_type = typename raster_type::shape_type;
    // for describing the layouts of tiles
    using layout_type = typename raster_type::layout_type;
    // for index arithmetic
    using index_type = typename raster_type::index_type;
    // for sizing things
    using size_type = typename raster_type::size_type;

    // adapter for tiles within my arena
    using tile_type = pyre::grid::grid_t<cell_type,
                                         layout_type,
                                         pyre::memory::view_t<cell_type>>;

    // interface
public:
    // add a reference tile to the pile
    inline void addReferenceTile(size_type pid, const constview_type & ref);
    // add a target search window to the pile
    inline void addTargetTile(size_type pid, const constview_type & tgt);

    // compute adjustments to the offset map
    inline auto adjust() -> const value_type *;

    // accessors
    inline auto pairs() const -> size_type;
    inline auto arena() const -> const cell_type *;

    // debugging support
    inline void dump() const;

    // meta-methods
public:
    inline ~Sequential();
    inline Sequential(size_type pairs,
                      const layout_type & refLayout, const layout_type & tgtLayout,
                      size_type refineFactor=2, size_type refineMargin=8,
                      size_type zoomFactor=4);

    // implementation details: methods
public:
    // push the tiles in the plan to device
    inline auto _push() const -> cell_type *;
    // compute the magnitude of the complex signal pixel-by-pixel
    inline auto _detect(const cell_type * cArena,
                        size_type refDim, size_type tgtDim) const -> value_type *;
    // subtract the mean from reference tiles and compute the square root of their variance
    inline auto _refStats(value_type * rArena,
                          size_type refDim, size_type tgtDim) const -> value_type *;
    // compute the sum area tables for the target tiles
    inline auto _sat(const value_type * rArena,
                     size_type refDim, size_type tgtDim) const -> value_type *;
    // compute the mean of all possible placements of a tile the same size as the reference
    // tile within the target
    inline auto _tgtStats(const value_type * sat,
                          size_type refDim, size_type tgtDim, size_type corDim
                          ) const -> value_type *;
    // correlate
    inline auto _correlate(const value_type * rArena,
                           const value_type * refStats, const value_type * tgtStats,
                           size_type refDim, size_type tgtDim, size_type corDim
                           ) const -> value_type *;
    // find the locations of the maxima of the correlation matrix
    inline auto _maxcor(const value_type * gamma, size_type corDim) const -> int *;
    // adjust the locations of the maxima so that the refined tile sources fit with the target
    inline void _nudge(int * locations, size_type refDim, size_type tgtDim) const;
    // allocate memory for a new arena big enough to hold the refined tiles
    inline auto _refinedArena() const -> cell_type *;
    // refine the reference tiles
    inline void _refRefine(cell_type * coarseArena, cell_type * refinedArena) const;
    // migrate the expanded unrefined target tiles into the {refinedArena}
    inline void _tgtMigrate(cell_type * coarseArena, int * locations,
                            cell_type * refinedArena) const;
    // refine the target tiles
    inline void _tgtRefine(cell_type * refinedArena) const;
    // deramp
    inline void _deramp(cell_type * arena) const;
    // zoom the correlation matrix
    inline auto _zoomcor(value_type * gamma) const -> value_type *;
    // assemble the offsets
    inline auto _offsetField(const int * maxcor, const int * zoomed) -> const value_type *;

    // unfinished correlation matrix zoom that uses R2C and C2R
    inline auto _zoomcor_r2r(value_type * gamma) const -> value_type *;


    // implementation details: data
private:
    // my capacity, in {ref/tgt} pairs
    const size_type _pairs;
    const size_type _refineFactor;
    const size_type _refineMargin;
    const size_type _zoomFactor;

    // the shape of the reference tiles
    const layout_type _refLayout;
    // the shape of the search windows in the target image
    const layout_type _tgtLayout;
    // the shape of the correlation matrix
    const layout_type _corLayout;
    // the shape of the reference tiles after refinement
    const layout_type _refRefinedLayout;
    // the shape of the target tiles after refinement
    const layout_type _tgtRefinedLayout;
    // the shape of the correlation matrix after refinement
    const layout_type _corRefinedLayout;
    // the shape of the correlation matrix after zooming
    const layout_type _corZoomedLayout;

    // the number of cells in a reference tile
    const size_type _refCells;
    // the number of cells in a target search window
    const size_type _tgtCells;
    // the number of cell in a correlation matrix
    const size_type _corCells;
    // the number of cells in a refined reference tile
    const size_type _refRefinedCells;
    // the number of cells in a refined target tile
    const size_type _tgtRefinedCells;

    // the number of bytes in a reference tile
    const size_type _refFootprint;
    // the number of bytes in a target search window
    const size_type _tgtFootprint;
    // the number of bytes in a correlation matrix
    const size_type _corFootprint;
    // the number of bytes in a refined reference tile
    const size_type _refRefinedFootprint;
    // the number of bytes in a refined target tile
    const size_type _tgtRefinedFootprint;

    // host storage for the tile pairs
    cell_type * const _arena;
    // host storage for the offset field
    value_type * const _offsets;
};


// code guard
#endif

// end of file
