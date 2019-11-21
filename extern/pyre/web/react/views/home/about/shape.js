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
# support
import pyre

# the component implementation
class Shape(pyre.protocol, family='gauss.shapes'):
    """
    The abstract specification of shape components
    """

    # interface requirements
    @pyre.provides
    def measure(self):
        """
        Compute my measure (length, area, volume, etc)
        """

    # supply a default implementation
    @classmethod
    def pyre_default(cls, **kwds):
        """
        The default {Shape} implementation
        """
        # pull {Ball}
        from .Ball import Ball
        # and make it the default shape
        return Ball

# end of file
`

// dress up
const content = ({...props}) => (
    <Python {...props}>
        {listing}
    </Python>
)

// publish
export default content

// end of file
