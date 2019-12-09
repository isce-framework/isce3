// -*- web -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// externals
import React from 'react'
// locals
import document from './styles'

// render
const Narrative = ({children, style}) => (
    <div style={{...document.narrative, ...style}}>
        {children}
    </div>
)

// defaults
Narrative.defaultProps = {
    style: {},
}

// publish
export default Narrative

// end of file
