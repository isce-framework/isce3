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
mga@cygnus:~> cd ~/tmp
mga@cygnus:~/tmp> bzr init pyre-1.0
mga@cygnus:~/tmp> cd pyre-1.0
mga@cygnus:~/tmp/pyre-1.0> bzr pull --remember http://pyre.orthologue.com/1.0/devel
    ...
All changes applied successfully
Now on revision xxxx
mga@cygnus:~/tmp/pyre-1.0>
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
