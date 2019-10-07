// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//


// for the build system
#include <portinfo>

// external packages
#include <map>
#include <vector>
#include <string>

// local types
#include "Device.h"
#include "Streaming.h"
#include "Renderer.h"
#include "Console.h"
#include "Chronicler.h"


// simplify access to the pyre::journal symbols
using namespace pyre::journal;


// initialize the default device
Chronicler::device_t * Chronicler::_defaultDevice = new Console();


// end of file
