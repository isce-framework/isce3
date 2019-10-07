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
mga@cygnus:~> cd ~/tmp/pyre-1.0
mga@cygnus:~/tmp/pyre-1.0> mm
 ...
mga@cygnus:~/tmp/pyre-1.0> mm.paths
mga@cygnus:~/tmp/pyre-1.0> python3
Python 3.6.5 (default, Mar 29 2018, 15:37:32)
[GCC 4.2.1 Compatible Apple LLVM 9.0.0 (clang-900.0.39.2)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import pyre
>>> pyre.__file__
'/Users/mga/tmp/pyre-1.0/products/packages/pyre/__init__.py'
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
