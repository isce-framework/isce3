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
mga@cygnus:~/tmp> bzr init config
mga@cygnus:~/tmp> cd config
mga@cygnus:~/tmp/config> bzr pull --remember http://config.orthologue.com/release
    ...
All changes applied successfully
Now on revision xxxx
mga@cygnus:~/tmp/config>
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
