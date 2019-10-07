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
# my protocol
from .Shape import Shape

# the component implementation
class Ball(pyre.component, family='gauss.shapes.ball', implements=Shape):
    """
    A shape that represents a sphere in {d} dimensions
    """

    # public state
    radius = pyre.properties.float(default=1)
    radius.doc = 'the radius of the ball'

    center = pyre.properties.array(default = (0,0))
    center.doc = 'the location of the center of the ball'

    # interface obligations
    @pyre.export
    def measure(self):
        """
        Compute my volume
        """
        # get functools and operator
        import functools, operator
        # get π
        from math import pi as π
        # compute the dimension of space
        d = len(self.center)
        # for even {d}
        if d % 2 == 0:
            # compute the scaling factor
            normalization = functools.reduce(operator.mul, range(1, d//2+1))
            # and assemble the volume
            return π**(d//2) * self.radius**d / normalization

        # for odd {d}, compute the scaling factor
        normalization = functools.reduce(operator.mul, range(1, d+1, 2))
        # and assemble the volume
        return 2**((d+1)//2) * π**((d-1)//2) / normalization

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
