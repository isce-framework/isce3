// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// A grid

// code guard
#if !defined(pyre_grid_Grid_h)
#define pyre_grid_Grid_h

// declaration
template <typename cellT, typename layoutT, typename storageT>
class pyre::grid::Grid {
    // types
public:
    // aliases for my template parameters
    using cell_type = cellT;
    using layout_type = layoutT;
    using storage_type = storageT;
    // view over portions of my data
    using view_type = View<Grid<cell_type, layout_type, storage_type>>;
    using constview_type = ConstView<Grid<cell_type, layout_type, storage_type>>;

    // dependent types
    using slice_type = typename layout_type::slice_type;
    using index_type = typename layout_type::index_type;
    using shape_type = typename layout_type::shape_type;
    using packing_type = typename layout_type::packing_type;

    // other help
    using size_type = std::size_t;

    // meta-methods
public:
    // given a layout and a storage solution managed by someone else
    inline Grid(layout_type layout, const storage_type & storage);
    // given a layout and a storage solution managed by me
    inline Grid(layout_type layout, storage_type && storage);
    // given a layout and a storage solution that can be instantiated using my shape info
    inline Grid(layout_type layout);
    // given just the index extents
    inline Grid(shape_type shape);

    // interface
public:
    // the dimensionality of my index
    inline static constexpr auto dim();

    // access to my shape
    inline const auto & layout() const;
    // access to my memory location
    inline auto data() const;
    // access to my storage strategy
    inline const auto & storage() const;

    // iteration support
    inline auto view();
    inline auto view(const slice_type & slice);

    inline auto view() const;
    inline auto view(const slice_type & slice) const;

    inline auto constview() const;
    inline auto constview(const slice_type & slice) const;

    // read and write access using offsets
    inline auto & operator[](size_type offset);
    inline auto & operator[](size_type offset) const;

    // read and write access using indices
    inline auto & operator[](const index_type & index);
    inline const auto & operator[](const index_type & index) const;

    // implementation details - data
private:
    const layout_type _layout;
    const storage_type _storage;
};


#endif

// end of file
