// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_grid_Slice_h)
#define pyre_grid_Slice_h


// declaration
template <typename indexT, typename packingT>
class pyre::grid::Slice {
    // types
public:
    // for sizing things
    typedef std::size_t size_type;
    // aliases for my parts
    typedef indexT index_type;
    typedef indexT shape_type;
    typedef packingT packing_type;
    // alias for me
    typedef Slice<index_type, packing_type> slice_type;
    // iterators
    typedef Iterator<slice_type> iterator_type;

    // meta-methods
public:
    // a slice is the set of indices [low, high) visited in a given packing
    Slice(const index_type & low, const index_type & high, const packing_type & packing);

    // interface
public:
    inline const auto & low() const;
    inline const auto & high() const;

    inline auto shape() const;
    inline const auto & packing() const;

    // iteration support
    inline auto begin() const;
    inline auto end() const;

    // implementation details
private:
    const index_type _low;
    const index_type _high;
    const packing_type _packing;
};


# endif

// end of file
