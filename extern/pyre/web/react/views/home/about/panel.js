// -*- web -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// externals
import React from 'react'

// support
import Python from 'widgets/code/Python'

// the code snippet
const listing = `\
# -*- coding: utf-8 -*-

# support
import isce

# the action implementation
class say(isce.panel(), family='isce.actions.say'):
    """
    An example of a custom action provided by the end user
    """

    # constants
    pyre_tip = 'an example of a custom action provided by the end user'

    # behavior
    @isce.export(tip='an example of a custom behavior')
    def hello(self, plexus, **kwds):
        """
        A custom behavior
        """
        # say something
        plexus.info.log("hello world!")
        # all done
        return 0

# end of file
`

// dress up
const content = () => (
    <Python>
        {listing}
    </Python>
)

// publish
export default content

// end of file
