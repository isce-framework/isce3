// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved


// code guard
#if !defined(pyre_version_h)
#define pyre_version_h

// support
#include <tuple>
#include <string>

// my declarations
namespace pyre {
    // my version is an array of three integers and the git hash
    using version_t = std::tuple<int, int, int, std::string>;

    // access to the version number of the {pyre} library
    version_t version();
}

#endif

// end of file
