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
#include <iostream>
#include <sstream>

// access to the low level diagnostic header file
#include <pyre/journal/Device.h>
#include <pyre/journal/Renderer.h>
#include <pyre/journal/Chronicler.h>
#include <pyre/journal/Diagnostic.h>
#include <pyre/journal/Index.h>
#include <pyre/journal/Inventory.h>
#include <pyre/journal/Channel.h>
#include <pyre/journal/macros.h>

#include <pyre/journal/Null.h>
#include <pyre/journal/Locator.h>
#include <pyre/journal/Selector.h>
#include <pyre/journal/manipulators.h>


// a simple channel class
class Debug :
    public pyre::journal::Diagnostic<Debug>,
    public pyre::journal::Channel<Debug, false>
{
    // befriend my superclass so it can invoke my recording hooks
    friend class Channel<Debug, false>;

    // types
public:
    using channel_t = Channel<Debug, false>;
    using index_t = channel_t::index_t;
    typedef std::string string_t;


    // meta methods
public:
    Debug(string_t name) :
        pyre::journal::Diagnostic<Debug>("debug", name),
        pyre::journal::Channel<Debug, false>(name)
    {}

    // per class
private:
    static index_t _index;
};

// allocate the index
Debug::index_t Debug::_index = Debug::index_t();

// main program
int main() {

    // instantiate
    Debug d("pyre.journal.test");
    // inject
    d << pyre::journal::Selector("key", "value");
    d << pyre::journal::Locator(__HERE__);
    d << "Hello world!";
    d << pyre::journal::newline;
    d << pyre::journal::endl;;

    // all done
    return 0;
}

// end of file
