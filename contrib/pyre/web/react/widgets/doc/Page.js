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
const Page = ({children, style}) => (
    <div style={{...document.page, ...style}}>
        {children}
    </div>
)

// defaults
Page.defaultProps = {
    style: {},
}

// publish
export default Page

// end of file
