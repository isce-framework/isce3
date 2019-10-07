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
# set up config
  alias mm='python3 \${HOME}/tmp/config/make/mm.py'
  alias mm.env='mm --env=sh'
  alias mm.show='mm --show --dry'
  alias mm.bldroot='mm --dry --quiet --show=BLD_ROOT'

  mm.paths() {
    # get {mm} to print out the path variables and add them to the current environment
    eval \$(python3 \${HOME}/tmp/config/make/mm.py --quiet --paths=sh \$*)
  }
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
