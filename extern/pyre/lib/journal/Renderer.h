// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_journal_Renderer_h)
#define pyre_journal_Renderer_h

// place Renderer in namespace pyre::journal
namespace pyre {
    namespace journal {
        class Renderer;
    }
}

// declaration
class pyre::journal::Renderer {
    // types
public:
    using string_t = std::string;
    using stream_t = std::stringstream;
    using entry_t = std::vector<string_t>;
    using metadata_t = std::map<string_t, string_t>;

    // interface
public:
    virtual string_t render(entry_t &, metadata_t &);

    // meta methods
public:
    virtual ~Renderer();
    inline Renderer();

    // disallow
private:
    Renderer(const Renderer &) = delete;
    const Renderer & operator=(const Renderer &) = delete;

    // implementation details
protected:
    virtual void header(stream_t &, entry_t &, metadata_t &);
    virtual void body(stream_t &, entry_t &, metadata_t &);
    virtual void footer(stream_t &, entry_t &, metadata_t &);
};


// get the inline definitions
#define pyre_journal_Renderer_icc
#include "Renderer.icc"
#undef pyre_journal_Renderer_icc


# endif
// end of file
