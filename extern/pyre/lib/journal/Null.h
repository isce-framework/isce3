// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(pyre_journal_Null_h)
#define pyre_journal_Null_h

// place Null in namespace pyre::journal
namespace pyre {
    namespace journal {
        class Null;

        // the injection operator
        template <typename item_t>
        inline Null & operator << (Null &, item_t);
    }
}

// declaration
class pyre::journal::Null
{
    // types
public:
    using string_t = std::string;

    // interface
public:
    // accessors
    inline bool isActive() const;

    // mutators
    inline void activate() const;
    inline void deactivate() const;

    // meta methods
public:
    inline operator bool() const;

    inline ~Null();
    inline Null(const string_t &);

    // disallow
private:
    Null(const Null &) = delete;
    const Null & operator=(const Null &) = delete;
};


// get the inline definitions
#define pyre_journal_Null_icc
#include "Null.icc"
#undef pyre_journal_Null_icc


# endif
// end of file
