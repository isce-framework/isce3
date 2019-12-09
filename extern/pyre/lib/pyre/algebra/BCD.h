// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

#if !defined(pyre_algebra_BCD_h)
#define pyre_algebra_BCD_h

// to get std::abs
#include <cstdlib>

namespace pyre {
    namespace algebra {
        template <size_t scale, typename precision_t=size_t> class BCD;
    }
}


// global arithmetic operators
// binary +
template <size_t scale, typename precision_t>
pyre::algebra::BCD<scale, precision_t>
operator+(
          const pyre::algebra::BCD<scale, precision_t> &,
          const pyre::algebra::BCD<scale, precision_t> &
          );


// binary -
template <size_t scale, typename precision_t>
pyre::algebra::BCD<scale, precision_t>
operator-(
          const pyre::algebra::BCD<scale, precision_t> &,
          const pyre::algebra::BCD<scale, precision_t> &
          );


// the BCD class
template <size_t scale, typename precision_t>
class pyre::algebra::BCD {
    // interface
public:

    // convert to double
    operator double () const;

    // arithmetic
    BCD operator+ () const;
    BCD operator- () const;

    BCD & operator+= (const BCD &);
    BCD & operator-= (const BCD &);

    // meta methods
public:
    inline ~BCD();

    inline BCD(precision_t msw=0, precision_t lsw=0);
    BCD(const BCD &);
    const BCD & operator= (const BCD &);

    // data members
public:
    precision_t _msw;
    precision_t _lsw;

    static const size_t _scale = scale;

};


// get the inline definitions
#define pyre_algebra_BCD_icc
#include "BCD.icc"
#undef pyre_algebra_BCD_icc


#endif

// end of file
