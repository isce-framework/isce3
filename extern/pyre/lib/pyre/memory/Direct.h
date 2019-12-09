// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_memory_Direct_h)
#define pyre_memory_Direct_h

//
// Direct is the life cycle manager of a memory mapping
//
// It is a low level concept and should be considered an implementation detail; as such, you
// should probably avoid using it directly
//

// declaration
template <typename cellT>
class pyre::memory::Direct : public pyre::memory::MemoryMap {
    // types
public:
    typedef cellT cell_type;
    typedef cell_type & reference;
    typedef const cell_type & const_reference;
    typedef cell_type * pointer;
    typedef const cell_type * const_pointer;

    // meta-methods
public:
    // constructor
    inline
    Direct(uri_type uri,         // the name of the file
           size_type size,       // how much of the file to map (in cell_type units)
           size_type offset = 0, // starting at this offset from the beginning
           bool preserve = false // preserve the file
           );

    // move semantics
    inline Direct(Direct &&) = default;
    inline Direct & operator=(Direct &&) = default;

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
    Direct() = delete;
    // copy semantics
    Direct(const Direct &) = delete;
    Direct & operator=(const Direct &) = delete;
};


#endif

// end of file
