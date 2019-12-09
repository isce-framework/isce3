// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_grid_Index_h)
#define pyre_grid_Index_h


// declaration
template <typename repT>
class pyre::grid::Index {
    // types
public:
    // the container of the index values
    typedef repT rep_type;
    // for sizing things
    typedef typename rep_type::size_type size_type;
    // just in case anybody cares about the type of my payload
    typedef typename rep_type::value_type value_type;

    // meta-methods
public:
    // the constructor is a variadic template to enable construction of the rep using
    // initializer lists; an alternative is to relax the access control to {_index} to allow the
    // compiler to do it
    template <typename... argT> inline Index(argT... value);

    // interface
public:
    // factories
    inline static constexpr auto fill(value_type value);

    // dimensionality
    inline static constexpr auto dim();
    // same as {dim} but as a member function
    inline constexpr auto size() const;

    // indexed access
    inline auto & operator[](size_type item);
    inline auto operator[](size_type item) const;

    // support for ranged for loops
    inline auto begin() const;
    inline auto end() const;

    inline auto begin();
    inline auto end();

    // implementation details
private:
    // there are two ways to initialize _index
    //  - relax access control to enable initialization through brace enclosed
    //    initializer lists
    //  - make the constructor a variadic template with a parameter pack
    // we picked the latter
    repT _index;
};


#endif

// end of file
