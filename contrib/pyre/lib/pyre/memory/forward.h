// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_memory_forward_h)
#define pyre_memory_forward_h

// externals
#include <stdexcept>
#include <fstream>
#include <utility>
// low level stuff
#include <cstring> // for strerror
#include <fcntl.h> // for open
#include <unistd.h> // for close
#include <sys/stat.h> // for the mode flags
#include <sys/mman.h> // for mmap
// support
#include <pyre/journal.h>

// forward declarations
namespace pyre {
    namespace memory {
        // local type aliases
        // for filenames
        typedef std::string uri_t;
        // for describing shapes and regions
        typedef off_t offset_t;
        typedef std::size_t size_t;
        // file information
        typedef struct stat info_t;

        class MemoryMap; // infrastructure

        // buffer types
        template <typename cellT> class View;         // view over existing memory
        template <typename cellT> class ConstView;    // view over existing constant memory
        template <typename cellT> class Heap;         // dynamically allocated memory
        template <typename cellT> class Direct;       // memory mapped file
        template <typename cellT> class ConstDirect;  // const access to a memory mapped file
    }
}

// type aliases for the above
namespace pyre {
    namespace memory {
        template <typename cellT> using view_t  = View<cellT>;
        template <typename cellT> using constview_t  = ConstView<cellT>;
        template <typename cellT> using heap_t  = Heap<cellT>;
        template <typename cellT> using direct_t  = Direct<cellT>;
        template <typename cellT> using constdirect_t  = ConstDirect<cellT>;
    }
}

#endif

// end of file
