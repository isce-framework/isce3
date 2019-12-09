// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_journal_Index_h)
#define pyre_journal_Index_h

// place Index in namespace pyre::journal
namespace pyre {
    namespace journal {
        template <typename> class Index;
    }
}

// This class maintains the map
//
//    channel name -> channel inventory
//
// to enable the maintenance of per-channel information. Each concrete subclass of {Diagnostic}
// maintains an {Index<Inventory<bool>>} to hold the activation state its output channels.

// declaration
template <typename Value>
class pyre::journal::Index {

    // types
public:
    using value_t = Value;
    using string_t = std::string;
    using key_t = string_t;
    using index_t = std::map<key_t, value_t>;

    // interface: place the public methods here
public:
    inline value_t & lookup(const string_t & channel);

    // meta methods: constructors, destructors
public:
    inline ~Index();
    inline Index();
    inline Index(const Index &);
    inline Index & operator=(const Index &);

    // data members
private:
    index_t _index;
};


// get the inline definitions
#define pyre_journal_Index_icc
#include "Index.icc"
#undef pyre_journal_Index_icc


# endif
// end of file
