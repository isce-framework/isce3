// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_memory_Heap_h)
#define pyre_memory_Heap_h

//
// Heap manages a dynamically allocated memory buffer
//
// It is a low level concept and should be considered an implementation detail; as such, you
// should probably avoid using it directly
//

// declaration
template <typename cellT>
class pyre::memory::Heap {
    // types
public:
    // aliases for my template type
    typedef cellT cell_type;
    typedef cell_type & reference;
    typedef const cell_type & const_reference;
    typedef cell_type * pointer;
    typedef const cell_type * const_pointer;
    // other help
    typedef size_t size_type;

    // meta-methods
public:
    // constructor
    inline Heap(size_type size);

    // move semantics
    inline Heap(Heap && other);
    inline Heap & operator=(Heap && other);

    // destructor
    inline ~Heap();

    // interface
public:
    // accessors
    inline size_type size() const;
    inline auto data() const;

    // implementation details: data
private:
    pointer _buffer;
    size_type _size;

    // suppress
private:
    // copy semantics
    Heap(const Heap &) = delete;
    Heap & operator=(const Heap &) = delete;

};


#endif

// end of file
