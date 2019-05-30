// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(ampcor_libampcor_correlators_correlator_h)
#define ampcor_libampcor_correlators_correlator_h


// externals
#include <pyre/grid.h>

// encapsulation of Correlator tables
template <typename rasterT>
class ampcor::correlators::Correlator {
    // types
public:
    // my client raster type
    using raster_type = rasterT;
    // its views
    using view_type = typename raster_type::view_type;

    // the payload type of my amplitude grids
    using pixel_type = float;
    // my grid type
    using grid_type = heapgrid_t<raster_type::layout_type::dim(), pixel_type>;
    // its index type
    using index_type = typename grid_type::index_type;

    // the sum area table of the amplitude of the target raster
    using sat_type = sumarea_t<grid_type>;

    // interface
public:
    // access to my amplitude grids
    const auto & correlation();

    // compute the correlation matrix of two matching tiles in the reference and target images
    const auto & correlate();

    // meta-methods
public:
    inline Correlator(const view_type & refView,
                      const view_type & tgtView
                      );

    // implementation details: data
private:
    // the quality grid
    grid_type _correlation;

    // the amplitude of the reference SLC
    grid_type _refAmplitudeSpread;
    // the variance of the amplitude of the reference SLC
    pixel_type _refAmplitudeVariance;  // the variance of the reference amplitude

    // the amplitude of the target SLC:
    grid_type _tgtAmplitude;
};


// code guard
#endif

// end of file
