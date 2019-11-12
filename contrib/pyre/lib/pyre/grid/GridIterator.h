// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_grid_GridIterator_h)
#define pyre_grid_GridIterator_h


// declaration
template <typename gridT>
class pyre::grid::GridIterator {
    // types
public:
    // my parts
    using grid_type = gridT;
    using slice_type = typename grid_type::slice_type;
    using iterator_type = typename slice_type::iterator_type;
    using value_type = typename grid_type::cell_type;

    // meta-methods
public:
    inline GridIterator(grid_type & grid, const iterator_type & it);

    // interface
public:
    inline auto & operator++();
    inline auto & operator*() const;

    // access to the iterator
    inline const auto & iterator() const;

    // implementation details
private:
    grid_type & _grid;
    iterator_type _iterator;
};


// GridIterator traits
// 20180726: switch to {std::} to keep gcc-6.4 happy; we need it for cuda
namespace std {
    template <typename gridT>
    class iterator_traits<pyre::grid::GridIterator<gridT>> {
        // types
    public:
        using value_type = typename gridT::cell_type;
        using difference_type = void;
        using pointer = typename gridT::cell_type *;
        using reference = typename gridT::cell_type &;
        using iterator_category = std::forward_iterator_tag;
    };
}


#endif

// end of file
