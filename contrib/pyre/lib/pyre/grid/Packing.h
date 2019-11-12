// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_grid_Packing_h)
#define pyre_grid_Packing_h


// declaration
template <pyre::grid::size_t dim>
class pyre::grid::Packing {
    // types
public:
    // the container with the index packing
    typedef std::array<size_t, dim> rep_type;
    // for sizing things
    typedef typename rep_type::size_type size_type;
    // the base type of my values
    typedef typename rep_type::value_type value_type;

    // meta-methods
public:
    // the constructor is a variadic template; it enables construction of the rep using
    // initializer lists
    template <typename... argT> inline Packing(argT... arg);

    // interface
public:
    // factories
    // c-like: last index varies the fastest
    inline static constexpr auto rowMajor();
    // and its alias
    inline static constexpr auto c();
    // fortran-like: first index varies the fastest
    inline static constexpr auto columnMajor();
    // and its alias
    inline static constexpr auto fortran();

    // size
    inline constexpr auto size() const;

    // indexed access
    inline auto operator[](size_type item) const;

    // support for ranged for loops
    inline auto begin() const;
    inline auto end() const;

    // implementation details
private:
    const rep_type _packing;
};


#endif

// end of file
