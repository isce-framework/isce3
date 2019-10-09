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
    typedef pyre::geometry::cloud_t<> cloud_t;
    // make a cloud
    cloud_t cloud(2);

    // add a couple of points
    cloud[0] = {0., 0., 1.};
    cloud[1] = {1., 0., 1.};

    // make a channel
    pyre::journal::debug_t channel("pyre.geometry");
    // show me
    channel << pyre::journal::at(__HERE__);
    // the entire cloud
    for (auto p : cloud) {
        channel << "point: (" << p << ")" << pyre::journal::newline;
    }
    // flush
    channel << pyre::journal::endl;

    // all done
    return 0;
}


// end of file
