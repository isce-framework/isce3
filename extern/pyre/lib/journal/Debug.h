// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_journal_Debug_h)
#define pyre_journal_Debug_h

// place Debug in namespace pyre::journal
namespace pyre {
    namespace journal {
        class Debug;
    }
}


// declaration
class pyre::journal::Debug :
    public pyre::journal::Diagnostic<Debug>,
    public pyre::journal::Channel<Debug, false>
{
    // befriend my superclass so it can access my index
    friend class Channel<Debug, false>;

    // types
public:
    using string_t = std::string;
    using diagnostic_t = Diagnostic<Debug>;
    using channel_t = Channel<Debug, false>;
    using index_t = channel_t::index_t;

    // meta methods
public:
    inline ~Debug();
    inline explicit Debug(string_t name);

    // disallow
private:
    Debug(const Debug &) = delete;
    const Debug & operator=(const Debug &) = delete;

    // per class
private:
    static index_t _index;
};


// get the inline definitions
#define pyre_journal_Debug_icc
#include "Debug.icc"
#undef pyre_journal_Debug_icc


# endif
// end of file
