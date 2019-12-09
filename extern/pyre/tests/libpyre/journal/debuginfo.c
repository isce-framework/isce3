/* -*- coding: utf-8 -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
*/


/* for the build system */
#include <portinfo.h>

/* access to the journal header file */
#include <pyre/journal/debuginfo.h>

/* main program */
int main() {
    /* the channel name */
    const char * channel = "pyre.journal.test";
    /* activate it */
    debuginfo_activate(channel);
    /* if the channel is active */
    if (debuginfo_active(channel)) {
        /* deactivate it */
        debuginfo_deactivate(channel);
    }
    /* say something */
    debuginfo_out(channel, __HERE__, "%s", "hello");

    /* all done */
    return 0;
}

/* end of file */
