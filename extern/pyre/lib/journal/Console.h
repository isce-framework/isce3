// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_journal_Console_h)
#define pyre_journal_Console_h

// place Console in namespace pyre::journal
namespace pyre {
    namespace journal {
        class Console;
    }
}


// declaration
class pyre::journal::Console : public pyre::journal::Streaming {
    // types
public:
    using renderer_t = Renderer;

    // interface
public:
    virtual void record(entry_t &, metadata_t &);

    // meta methods
public:
    virtual ~Console();
    Console();

    // disallow
private:
    Console(const Console &) = delete;
    const Console & operator=(const Console &) = delete;

    // data
private:
    renderer_t * _renderer;
};


# endif
// end of file
