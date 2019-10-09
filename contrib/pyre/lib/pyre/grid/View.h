// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_grid_View_h)
#define pyre_grid_View_h


// declaration
template <typename gridT>
class pyre::grid::View {
    // types
public:
    // alias for my template parameter
    using grid_type = gridT;
    // and some of its parts
    using layout_type = typename grid_type::layout_type;
    using slice_type = typename grid_type::slice_type;
    // my iterator
    using iterator_type = GridIterator<grid_type>;

    // meta-methods
public:
    inline View(grid_type & grid, const slice_type & slice);
    inline const View & operator=(const View & view) const;

    // interface
public:
    inline const auto & low() const;
    inline const auto & high() const;
    inline const auto & packing() const;

    // compute my layout
    inline auto layout() const;
    // grant access to my slice
    inline const auto & slice() const;

    // iteration support
    inline auto begin() const;
    inline auto end() const;
    
    // implementation details
private:
    grid_type & _grid;
    slice_type _slice;
};


#endif

// end of file
