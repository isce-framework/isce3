// -*- web -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// externals
import React from 'react'

// support
import Bash from 'widgets/code/Bash'

// the code snippet
const listing = `\
mga@cygnus:~/tmp/pyre-1.0> cd ~
mga@cygnus:~> mkdir .mm
mga@cygnus:~> cd .mm
mga@cygnus:~/.mm> vi mga.py
mga@cygnus:~/.mm>
`

// dress up
const content = ({...props}) => (
    <Bash {...props}>
        {listing}
    </Bash>
)

// publish
export default content

// end of file
