// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// build system
#include <portinfo>

// system includes
#include <cassert>

// dependencies
#include <pyre/algebra/BCD.h>

// make short alias for the BCD type we are testing
typedef pyre::algebra::BCD<10> bcd;


// main program
int main(int argc, char* argv[]) {

    // default constructor
    bcd zero;
    assert(zero == 0);

    // constructor with explicit arguments
    bcd one(1,0);
    assert(one == 1);

    // copy constructor
    bcd copy(one);
    assert(copy == one);

    // operator =
    bcd another = one;
    assert(copy == one);

    // operator +
    assert(zero + zero == 0.0);
    assert(one + zero == 1.0);
    assert(zero + one == 1.0);
    assert(one + one == 2.0);

    // operator +=
    another += zero;
    assert(another == 1);
    another += one;
    assert(another == 2);

    //  operator -=
    another -= one;
    assert(another == 1);

    // exercise the overflow logic
    bcd almost(0,9);
    assert(almost + almost == 1.8);

    // all done
    return 0;
}


// end of file
