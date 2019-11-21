// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//


// for the build system
#include <portinfo>

// packages
#include <map>
#include <vector>
#include <string>
#include <iostream>

// local types
#include "Renderer.h"
#include "Device.h"
#include "Streaming.h"
#include "Console.h"


// interface
// record a diagnostic
void
pyre::journal::Console::
record(entry_t & entry, metadata_t & metadata)
{
    // get the renderer to convert the diagnostic parts into a string
    string_t text = _renderer->render(entry, metadata);
    // print it out
    std::cout << text;
    // and return
    return;
}


// destructor
pyre::journal::Console::
~Console()
{
    delete _renderer;
}


// the default constructor
pyre::journal::Console::
Console() :
    pyre::journal::Streaming(std::cout),
    _renderer(new Renderer())
{}


// end of file
