// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_grid_ConstView_h)
#define pyre_grid_ConstView_h


// declaration
template <typename gridT>
class pyre::grid::ConstView {
    // types
public:
    // alias for my template parameter
    using grid_type = gridT;
    // and some of its parts
    using layout_type = typename grid_type::layout_type;
    using slice_type = typename grid_type::slice_type;
    // my iterator
    using iterator_type = ConstGridIterator<grid_type>;

    // meta-methods
public:
    inline ConstView(const grid_type & grid, const slice_type & slice);

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
    const grid_type & _grid;
    slice_type _slice;
};


#endif

// end of file
