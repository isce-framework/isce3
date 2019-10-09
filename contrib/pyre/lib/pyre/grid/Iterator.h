// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_grid_Iterator_h)
#define pyre_grid_Iterator_h


// declaration
template <typename sliceT>
class pyre::grid::Iterator {
    // types
public:
    // my parts
    using slice_type = sliceT;
    using index_type = typename slice_type::index_type;
    using packing_type = typename slice_type::packing_type;

    // meta-methods
public:
    inline Iterator(const slice_type & slice);
    inline Iterator(const index_type & current, const slice_type & slice);

    // interface
public:
    inline auto & operator++();
    inline const auto & operator*() const;

    // implementation details
private:
    index_type _current;
    const slice_type & _slice;
};


// Iterator traits
// 20180726: switch to {std::} to keep gcc-6.4 happy; we need it for cuda
namespace std {
    template <typename sliceT>
    class iterator_traits<pyre::grid::Iterator<sliceT>> {
        // types
    public:
        using value_type = typename sliceT::index_type;
        using difference_type = void;
        using pointer = typename sliceT::cell_type *;
        using reference = typename sliceT::cell_type &;
        using iterator_category = std::forward_iterator_tag;
    };
}


#endif

// end of file
