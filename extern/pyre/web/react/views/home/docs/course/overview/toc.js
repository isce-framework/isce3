// -*- web -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// externals
import React from 'react'
import { Link } from 'react-router-dom'

// support
import { Pyre, TableOfContents, Content } from 'widgets/doc'

// locals
import styles from './styles'

// declaration
const TOC = () => (
    <TableOfContents title="Table of Contents">
        <Link to="/docs/course/overview/intro">introduction</Link>
        <Link to="/docs/course/overview/components">components</Link>
        <Link to="/docs/course/overview/appications">applications</Link>
        <Link to="/docs/course/overview/appications">persistence</Link>
    </TableOfContents>
)

// publish
export default TOC

// end of file
