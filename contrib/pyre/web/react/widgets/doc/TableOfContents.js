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
const TableOfContents = ({title, children}) => (
    <section style={document.toc.container}>
        <span style={document.toc.title}>{title}</span>
        {children}
    </section>
)

// defaults
TableOfContents.defaultProps = {
    title: "Contents",
}

// publish
export default TableOfContents

// end of file
