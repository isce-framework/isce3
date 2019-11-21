// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_grid_Layout_h)
#define pyre_grid_Layout_h

// declaration
template <typename indexT, typename packingT>
class pyre::grid::Layout : public Slice<indexT, packingT> {
    // types
public:
    // for sizing things
    typedef std::size_t size_type;
    // aliases for my parts
    typedef indexT index_type;
    typedef indexT shape_type;
    typedef packingT packing_type;
    typedef Slice<indexT, packingT> slice_type;

    // meta-methods
public:
    // a layout with index packinging supplied by the caller
    Layout(shape_type shape, packing_type packing = packing_type::rowMajor());

    // interface
public:
    // the dimensionality of my index
    inline static constexpr auto dim();
    // the number of cells in this layout
    inline auto size() const;

    // compute the pixel offset implied by a given index
    inline auto offset(const index_type & index) const;
    // compute the index that corresponds to a given offset
    inline auto index(size_type offset) const;

    // syntactic sugar for the pair above
    inline auto operator[](const index_type & index) const;
    inline auto operator[](size_type offset) const;

    // iterating over slices in arbitrary packing order
    auto slice(const packing_type & packing) const;
    auto slice(const index_type & begin, const index_type & end) const;
    auto slice(const index_type & begin, const index_type & end,
               const packing_type & packing) const;
};


#endif

// end of file
