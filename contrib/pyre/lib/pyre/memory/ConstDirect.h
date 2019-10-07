// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_memory_ConstDirect_h)
#define pyre_memory_ConstDirect_h

//
// ConstDirect is the life cycle manager of a memory mapping
//
// It is a low level concept and should be considered an implementation detail; as such, you
// should probably avoid using it directly
//

// declaration
template <typename cellT>
class pyre::memory::ConstDirect : public pyre::memory::MemoryMap {
    // types
public:
    typedef cellT cell_type;
    typedef const cell_type & reference;
    typedef const cell_type & const_reference;
    typedef const cell_type * pointer;
    typedef const cell_type * const_pointer;

    // meta-methods
public:
    // constructor
    inline
    ConstDirect(uri_type uri,         // the name of the file
                size_type size,       // how much of the file to map (in cell_type units)
                size_type offset = 0, // starting at this offset from the beginning
                bool preserve = false // preserve the file
                );

    // move semantics
    inline ConstDirect(ConstDirect &&) = default;
    inline ConstDirect & operator=(ConstDirect &&) = delete;

    // interface
public:
    // accessors
    inline auto size() const;
    inline auto data() const;

    // implementation details: data
private:
    size_type _size;

    // suppress
private:
    // default constructor
    ConstDirect() = delete;
    // copy semantics
    ConstDirect(const ConstDirect &) = delete;
    ConstDirect & operator=(const ConstDirect &) = delete;
};


#endif

// end of file
