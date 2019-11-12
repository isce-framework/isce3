// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//


// declaration of points
template <std::size_t dim, typename dataT>
class pyre::geometry::Point {
    // types
public:
    typedef dataT data_type;
    typedef std::size_t size_type;
    typedef std::array<data_type, dim> rep_type;

    // meta-methods
public:
    template <typename... coordT> inline Point(coordT... coordinate);

    // interface
public:
    // the dimension of the underlying space
    inline constexpr static auto dimension();
    // access to my contents
    const auto & data() const;

    // indexed access
    inline auto operator[](size_type item) const;

    // support for ranged for loops
    inline auto begin() const;
    inline auto end() const;

    // implementation details
private:
    rep_type _coordinates;
};


// end of file
