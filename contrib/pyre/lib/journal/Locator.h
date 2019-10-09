// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//


#if !defined(pyre_journal_Locator_h)
#define pyre_journal_Locator_h


// forward declarations
namespace pyre {
    namespace journal {
        class Locator;

        // overload the injection operator
        template <typename Channel>
        inline
        Diagnostic<Channel> &
        operator << (Diagnostic<Channel> &, const Locator &);
    }
}

// null diagnostics
inline
pyre::journal::Null &
operator<< (pyre::journal::Null &, const pyre::journal::Locator &);


// locator
class pyre::journal::Locator {
    // interface
public:
    template <typename Channel>
    inline
    Diagnostic<Channel> &
    inject(Diagnostic<Channel> & channel) const;

    // meta methods
public:
    inline ~Locator();
    inline Locator(const char *, int, const char * = 0);
    inline Locator(const Locator &);
    // disabled
private:
    inline Locator & operator=(const Locator &);

    // data
private:
    const char * _file;
    int _line;
    const char * _function;
};


// get the inline definitions
#define pyre_journal_Locator_icc
#include "Locator.icc"
#undef pyre_journal_Locator_icc

#endif // pyre_journal_manipulators_0_h

// end of file
