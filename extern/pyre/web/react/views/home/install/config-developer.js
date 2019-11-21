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
import os

# adjust the {builder} with {developer} choices
def developer(builder):
    """
    Decorate the builder with developer specific choices
    """
    # get the developer name
    user = builder.user
    # place temporary build files in my 'tmp' directory
    builder.bldroot = os.path.join(user.home, 'tmp', 'builds')
    # and the products in my 'tools' directory
    builder.prefix = os.path.join(user.home, 'tools')

    # all done
    return builder
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
