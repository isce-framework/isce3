// -*- web -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// externals
import React from 'react'
// locals
import document  from './styles'

// render
const Paragraph = ({children, style}) => (
    <p style={{...document.body, ...style}}>
        {children}
    </p>
)

// publish
export default Paragraph

// end of file
