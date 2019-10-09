// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_grid_DirectGrid_h)
#define pyre_grid_DirectGrid_h

// A memory mapped grid

// declaration
template < typename cellT, typename layoutT, typename directT>
class pyre::grid::DirectGrid : public Grid<cellT, layoutT, directT> {
    // types
public:
    // aliases for my template parameters
    typedef cellT cell_type;
    typedef layoutT layout_type;
    typedef directT storage_type;
    // dependent types
    typedef typename storage_type::uri_type uri_type;
    typedef typename layout_type::index_type index_type;
    typedef typename layout_type::packing_type packing_type;

    // other help
    typedef std::size_t size_type;

    // meta-methods
public:
    inline DirectGrid(uri_type uri, layout_type shape);
};


# endif

// end of file
