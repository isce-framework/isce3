/* -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
*/


/* for the build system */
#include <portinfo.h>

/* access to the journal header file */
#include <pyre/journal/firewalls.h>

/* main program */
int main() {

    /* the channel name */
    const char * channel = "pyre.journal.test";

    /* check a trivial condition */
    firewall_check(channel, 0==0, __HERE__, "%s", "oooooops!");

    // all done
    return 0;
}

/* end of file */
