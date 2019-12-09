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

// access to the low level inventory header file
#include <pyre/journal/Inventory.h>
#include <pyre/journal/Index.h>
#include <pyre/journal/Channel.h>

using namespace pyre::journal;

// convenience
typedef Inventory<true> true_t;
typedef Inventory<false> false_t;

// must subclass since the Channel constructor and destructor are protected
class trueref_t : public Channel<trueref_t> {
    // befriend my superclass so it can invoke my recording hooks
    friend class Channel<trueref_t>;
    // types
public:
    using channel_t = Channel<trueref_t>;
    using index_t = channel_t::index_t;
public:
    trueref_t(string_t name) : Channel<trueref_t>::Channel(name) {}
    // per class
private:
    static index_t _index;
};


class falseref_t : public Channel<falseref_t, false> {
    // befriend my superclass so it can invoke my recording hooks
    friend class Channel<falseref_t,false>;
    // types
public:
    using channel_t = Channel<falseref_t, false>;
    using index_t = channel_t::index_t;
public:
    falseref_t(string_t name) : Channel<falseref_t,false>::Channel(name) {}
// per class
private:
    static index_t _index;
};

// declare the indices
trueref_t::index_t trueref_t::_index = trueref_t::index_t();
falseref_t::index_t falseref_t::_index = falseref_t::index_t();

// main program
int main() {

    // and wrap channels over them
    trueref_t on_ref("true");
    falseref_t off_ref("true");

    // check their default settings
    assert(on_ref.isActive() == true);
    assert(off_ref.isActive() == false);

    // flip them
    on_ref.deactivate();
    off_ref.activate();

    // check again
    assert(on_ref.isActive() == false);
    assert(off_ref.isActive() == true);

    // all done
    return 0;
}

// end of file
