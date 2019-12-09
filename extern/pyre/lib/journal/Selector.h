// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//


#if !defined(pyre_journal_Selector_h)
#define pyre_journal_Selector_h


// forward declarations
namespace pyre {
    namespace journal {
        class Selector;

        // the injection operator
        template <typename Channel>
        inline
        Diagnostic<Channel> &
        operator << (Diagnostic<Channel> &, const Selector &);
    }
}


// null diagnostics
inline
pyre::journal::Null &
operator<< (pyre::journal::Null &, const pyre::journal::Selector &);


// definitions
// attributes
class pyre::journal::Selector {
    // types
public:
    using string_t = std::string;

    // interface
public:
    template <typename Channel>
    inline
    Diagnostic<Channel> &
    inject(Diagnostic<Channel> & channel) const;

    // meta methods
public:
    inline ~Selector();
    inline Selector(string_t, string_t);
    // inline Selector(const Selector &); // accessible but not implemented; for C++89 compatibility

    // disallow
private:
    Selector(const Selector &) = delete;
    Selector & operator= (const Selector &) = delete;

    // data
private:
    string_t _key;
    string_t _value;
};


// get the inline definitions
#define pyre_journal_Selector_icc
#include "Selector.icc"
#undef pyre_journal_Selector_icc

#endif // pyre_journal_manipulators_0_h

// end of file
