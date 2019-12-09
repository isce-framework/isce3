// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_journal_Informational_h)
#define pyre_journal_Informational_h

// place Informational in namespace pyre::journal
namespace pyre {
    namespace journal {
        class Informational;
    }
}


// declaration
class pyre::journal::Informational :
    public pyre::journal::Diagnostic<Informational>,
    public pyre::journal::Channel<Informational, true>
{
    // befriend my superclass so it can access my index
    friend class Channel<Informational, true>;

    // types
public:
    using string_t = std::string;
    using diagnostic_t = Diagnostic<Informational>;
    using channel_t = Channel<Informational, true>;
    using index_t = channel_t::index_t;

    // meta methods
public:
    inline ~Informational();
    inline explicit Informational(string_t name);

    // disallow
private:
    Informational(const Informational &) = delete;
    const Informational & operator=(const Informational &) = delete;

    // per class
private:
    static index_t _index;
};


// get the inline definitions
#define pyre_journal_Informational_icc
#include "Informational.icc"
#undef pyre_journal_Informational_icc


# endif
// end of file
