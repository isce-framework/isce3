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
#include <map>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

// access to the low level diagnostic header file
#include <pyre/journal/Device.h>
#include <pyre/journal/Renderer.h>
#include <pyre/journal/Chronicler.h>
#include <pyre/journal/Diagnostic.h>

class Debug : public pyre::journal::Diagnostic<Debug> {
    // types
public:
    typedef std::string string_t;
    // meta methods
public:
    Debug(string_t name) : pyre::journal::Diagnostic<Debug>("debug", name) {}
};


// main program
int main() {

    // instantiate
    Debug d("pyre.journal.test");

    // all done
    return 0;
}

// end of file
