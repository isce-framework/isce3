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
#! /usr/bin/env python3
# support
import pyre

# the application class
class Gauss(pyre.application, family='gauss.applications.simple'):
    """
    A simple application that computes the volume of a shape
    """

    # public state
    shape = Shape()
    shape.doc = 'the shape whose volume will be computed'

    # interface
    @pyre.export
    def main(self):
        """
        Compute the volume of my shape
        """
        # ask the shape to measure itself
        measure = self.shape.measure()
        # display the result
        self.info.log("volume = {measure}".format(measure=measure))
        # report succecss
        return 0

# main
if __name__ == "__main__":
    # instantiate an app
    gauss = Gauss(name="simple")
    # invoke it
    status = gauss.run()
    # share the status with the user's shell
    raise SystemExit(status)

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
