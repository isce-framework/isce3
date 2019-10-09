// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_memory_ConstView_h)
#define pyre_memory_ConstView_h

//
// ConstView converts memory owned by some other object into a storage strategy
//
// It is a low level concept and should be considered an implementation detail; as such, you
// should probably avoid using it directly
//

// declaration
template <typename cellT>
class pyre::memory::ConstView {
    // types
public:
    typedef cellT cell_type;
    typedef const cell_type & reference;
    typedef const cell_type & const_reference;
    typedef const cell_type * pointer;
    typedef const cell_type * const_pointer;

    // meta-methods
public:
    // destructor
    inline ~ConstView();

    // constructor
    inline ConstView(const pointer buffer);

    // copy semantics
    inline ConstView(const ConstView & other);
    inline ConstView & operator=(const ConstView & other);

    // move semantics
    inline ConstView(const ConstView &&);
    inline ConstView & operator=(const ConstView &&);

    // interface
public:
    // accessor
    inline auto data() const;

    // implementation details: data
private:
    pointer const _buffer;
};


#endif

// end of file
