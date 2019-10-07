// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_journal_Firewall_h)
#define pyre_journal_Firewall_h

// place Firewall in namespace pyre::journal
namespace pyre {
    namespace journal {
        class Firewall;
    }
}


// declaration
class pyre::journal::Firewall :
    public pyre::journal::Diagnostic<Firewall>,
    public pyre::journal::Channel<Firewall, true>
{
    // befriend my superclass so it can invoke my recording hooks
    friend class Diagnostic<Firewall>;
    // befriend my superclass so it can access my index
    friend class Channel<Firewall, true>;

    // types
public:
    using string_t = std::string;
    using diagnostic_t = Diagnostic<Firewall>;
    using channel_t = Channel<Firewall, true>;
    using index_t = channel_t::index_t;

    // meta methods
public:
    inline ~Firewall();
    inline explicit Firewall(string_t name);

    // disallow
private:
    Firewall(const Firewall &) = delete;
    const Firewall & operator=(const Firewall &) = delete;

    // implementation details
protected:
    inline void _endRecording(); // NYI: are firewalls fatal?

    // per class
private:
    static index_t _index;
};


// get the inline definitions
#define pyre_journal_Firewall_icc
#include "Firewall.icc"
#undef pyre_journal_Firewall_icc


# endif
// end of file
