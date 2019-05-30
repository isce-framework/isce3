// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(ampcor_libampcor_dom_slc_h)
#define ampcor_libampcor_dom_slc_h


// externals
#include <complex>

// encapsulation of SLC raster images
class ampcor::dom::SLC : public ampcor::dom::Raster {
    // types
public:
    // for sizing things
    using size_type = size_t; // from ampcor::dom
    // filenames
    using uri_type = uri_t;
    // my cell type
    using cell_type = std::complex<float>;
    // my pixel
    using pixel_type = cell_type;
    // my grid
    using grid_type = constmmap_t<2, pixel_type>;
    // my layout type
    using layout_type = grid_type::layout_type;
    // my slice type
    using slice_type = grid_type::slice_type;
    // my index type
    using index_type = grid_type::index_type;
    // my shape type
    using shape_type = grid_type::shape_type;
    // my view type
    using view_type = grid_type::view_type;
    // my constview type
    using constview_type = grid_type::constview_type;

    // interface
public:
    // access to my layout
    inline auto layout() const;

    // the number of pixels
    inline auto pixels() const;
    // my memory footprint
    inline auto size() const;
    // the size of my pixel
    inline static constexpr auto pixelSize();

    // the location of the data buffer
    inline auto data() const;

    // data access
    inline const auto & operator[](size_type offset) const;
    inline const auto & operator[](const index_type & index) const;

    // support for iterating over my data
    inline auto view();
    inline auto view(const slice_type & slice);

    inline auto view() const;
    inline auto view(const slice_type & slice) const;

    inline auto constview(const slice_type & slice) const;

    // meta-methods
public:
    virtual ~SLC();
    inline SLC(uri_type filename, shape_type shape);

    // implementation details: data
private:
    grid_type _grid;
};


// code guard
#endif

// end of file
