// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_journal_Warning_h)
#define pyre_journal_Warning_h

// place Warning in namespace pyre::journal
namespace pyre {
    namespace journal {
        class Warning;
    }
}


// declaration
class pyre::journal::Warning :
    public pyre::journal::Diagnostic<Warning>,
    public pyre::journal::Channel<Warning, true>
{
    // befriend my superclass so it can access my index
    friend class Channel<Warning, true>;

    // types
public:
    using string_t = std::string;
    using diagnostic_t = Diagnostic<Warning>;
    using channel_t = Channel<Warning, true>;
    using index_t = channel_t::index_t;

    // meta methods
public:
    inline ~Warning();
    inline explicit Warning(string_t name);

    // disallow
private:
    Warning(const Warning &) = delete;
    const Warning & operator=(const Warning &) = delete;

    // per class
private:
    static index_t _index;
};


// get the inline definitions
#define pyre_journal_Warning_icc
#include "Warning.icc"
#undef pyre_journal_Warning_icc


# endif
// end of file
