// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_memory_View_h)
#define pyre_memory_View_h

//
// View converts memory owned by some other object into a storage strategy
//
// It is a low level concept and should be considered an implementation detail; as such, you
// should probably avoid using it directly
//

// declaration
template <typename cellT>
class pyre::memory::View {
    // types
public:
    typedef cellT cell_type;
    typedef cell_type & reference;
    typedef const cell_type & const_reference;
    typedef cell_type * pointer;
    typedef const cell_type * const_pointer;

    // meta-methods
public:
    // destructor
    inline ~View();

    // constructor
    inline View(cell_type * buffer = 0);

    // copy semantics
    inline View(const View & other);
    inline View & operator=(const View & other);

    // move semantics
    inline View(const View &&);
    inline View & operator=(const View &&);

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
