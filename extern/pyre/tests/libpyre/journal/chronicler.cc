// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//


// for the build system
#include <portinfo>

// packages
#include <cassert>
// access to the packages used by Chronicler
#include <map>
#include <vector>
#include <string>
#include <cstdlib>

// access to the low level header files
#include <pyre/journal/Index.h>
#include <pyre/journal/Device.h>
#include <pyre/journal/Renderer.h>
#include <pyre/journal/Chronicler.h>

// must subclass chronicler to gain access to its constructor/destructor, which are protected
class chronicler_t : pyre::journal::Chronicler {
public:
    ~chronicler_t() {}
    chronicler_t() {}
};

// main program
int main() {

    // instantiate a chronicler
    chronicler_t chronicler;

    // all done
    return 0;
}

// end of file
