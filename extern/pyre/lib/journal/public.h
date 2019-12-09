// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//


#if !defined(pyre_journal_public_h)
#define pyre_journal_public_h

// external packages
#include <map>
#include <vector>
#include <string>
#include <sstream>

// local declarations
// infrastructure
#include "Device.h"
#include "Chronicler.h"
#include "Inventory.h"
#include "Index.h"
#include "Channel.h"
#include "Diagnostic.h"
// the predefined diagnostics
#include "Debug.h"
#include "Error.h"
#include "Firewall.h"
#include "Informational.h"
#include "Null.h"
#include "Warning.h"
// manipulators and associated support
#include "macros.h"
#include "Locator.h"
#include "Selector.h"
#include "manipulators.h"

// typedefs for convenience
// debugging support
namespace pyre {
    namespace journal {

        //  if we are building the library
#if defined(PYRE_CORE)
        // we need everything
        using debug_t = Debug;
        using firewall_t = Firewall;

        // if this a client DEBUG build
#elif defined(DEBUG)
        // enable debug and firewalls
        using debug_t = Debug;
        using firewall_t = Firewall;

        // otherwise, this is a production build
#else
        // disable debug and firewalls for production builds
        using debug_t = Null;
        using firewall_t = Null;
#endif
    }
}

// diagnostics
namespace pyre {
    namespace journal {
        // diagnostics
        using error_t = Error;
        using info_t = Informational;
        using warning_t = Warning;

        // locators
        using at = Locator;
        using set = Selector;
    }
}

#endif // pyre_journal_h

// end of file
