// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_memory_MemoryMap_h)
#define pyre_memory_MemoryMap_h

// declaration
// this class is a wrapper around the os calls
//
// ALL SIZES HERE ARE IN BYTES
//
class pyre::memory::MemoryMap {
    // types
public:
    typedef pyre::memory::uri_t uri_type;
    typedef pyre::memory::info_t info_type;
    typedef pyre::memory::size_t size_type;
    typedef pyre::memory::offset_t offset_type;

    typedef void * pointer;

    // constants
public:
    constexpr static int entireFile = 0;

    // meta-methods
public:
    inline ~MemoryMap();

    // constructor
    MemoryMap(uri_type name, bool writable, size_type bytes, size_type offset, bool preserve);

    // move semantics
    inline MemoryMap(MemoryMap &&);
    inline MemoryMap & operator=(MemoryMap &&);

    // interface
public:
    inline auto uri() const;
    inline auto bytes() const;
    inline auto buffer() const;
    inline const auto & fileinfo() const;

    // implementation details - data
private:
    uri_type _uri;
    info_type _info;
    size_type _bytes;
    pointer _buffer;

    // class methods
public:
    static size_type create(uri_type name, size_type bytes);
    static pointer map(uri_type name, size_type bytes, size_type offset, bool writable);
    static void unmap(const pointer buffer, size_type bytes);

    // suppress
private:
    // copy semantics
    MemoryMap() = delete;
    MemoryMap(const MemoryMap &) = delete;
    MemoryMap & operator=(const MemoryMap &) = delete;
};

#endif

// end of file
