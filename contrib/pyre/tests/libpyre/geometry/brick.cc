// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// config
#include <portinfo>
// externals
#include <iostream>
#include <pyre/journal.h>
#include <pyre/geometry.h>

// main
int main() {
    // type aliases
    typedef pyre::geometry::point_t<3, double> point3d_t;
    typedef pyre::geometry::brick_t<3, point3d_t> brick3d_t;
    // make a bunch of points
    point3d_t p000 { 0., 0., 0. };
    point3d_t p100 { 1., 0., 0. };
    point3d_t p010 { 0., 1., 0. };
    point3d_t p110 { 1., 1., 0. };
    point3d_t p001 { 0., 0., 1. };
    point3d_t p101 { 1., 0., 1. };
    point3d_t p011 { 0., 1., 1. };
    point3d_t p111 { 1., 1., 1. };

    brick3d_t cube { p000, p100, p110, p010, p001, p101, p111, p011 };

    // make a channel
    pyre::journal::debug_t info("pyre.geometry");
    // show me
    info
        << pyre::journal::at(__HERE__)
        << "[" << cube << "]"
        << pyre::journal::endl;

    // all done
    return 0;
}


// end of file
