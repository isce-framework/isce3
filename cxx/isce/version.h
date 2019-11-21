// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Bryan V. Riel
// Copyright 2017-2018

// code guard
#if !defined(isce_version_h)
#define isce_version_h

// support
#include <array>

// my declarations
namespace isce {
    // my version is an array of three integers
    typedef std::array<int, 3> version_t;

    // access to the version number of the {pyre} library
    version_t version();
}

#endif

// end of file
