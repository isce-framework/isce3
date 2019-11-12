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
const Sidebar = ({children, style}) => (
    <div style={{...document.sidebar, ...style}}>
        {children}
    </div>
)

// defaults
Sidebar.defaultProps = {
    style: {},
}

// publish
export default Sidebar

// end of file
