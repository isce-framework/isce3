// -*- C++ -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// config
#include <portinfo>
// externals
#include <pyre/journal.h>
#include <pyre/geometry.h>

// main
int main() {
    // type alias
    typedef pyre::geometry::point_t<3, double> point3d_t;
    // make a point
    point3d_t p {0., 0., 0.};

    // make a channel
    pyre::journal::debug_t info("pyre.geometry");
    // show me
    info
        << pyre::journal::at(__HERE__)
        << "point: (" << p << ")"
        << pyre::journal::endl;

    // all done
    return 0;
}


// end of file
