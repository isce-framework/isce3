// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_journal_Error_h)
#define pyre_journal_Error_h

// place Error in namespace pyre::journal
namespace pyre {
    namespace journal {
        class Error;
    }
}


// declaration
class pyre::journal::Error :
    public pyre::journal::Diagnostic<Error>,
    public pyre::journal::Channel<Error, true>
{
    // befriend my superclass so it can access my index
    friend class Channel<Error, true>;

    // types
public:
    using string_t = std::string;
    using diagnostic_t = Diagnostic<Error>;
    using channel_t = Channel<Error, true>;
    using index_t = channel_t::index_t;

    // meta methods
public:
    inline ~Error();
    inline explicit Error(string_t name);

    // disallow
private:
    Error(const Error &) = delete;
    const Error & operator=(const Error &) = delete;

    // per class
private:
    static index_t _index;
};


// get the inline definitions
#define pyre_journal_Error_icc
#include "Error.icc"
#undef pyre_journal_Error_icc


# endif
// end of file
