// -*- web -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// externals
import React from 'react'

// support
import { Chapter, Narrative, Page, Pyre, Sidebar } from 'widgets/doc'

// locals
import styles from './styles'
import TOC from './toc'
import Prologue from './prologue'
import Introduction from './introduction'

// declaration
const Overview = () => (
    <Page>
        <Sidebar>
            <TOC/>
        </Sidebar>
        <Narrative>
            <Chapter id="overview"
                     style={styles}
                     title={<span>Overview of <Pyre/></span>}>
                <Prologue/>
                <Introduction/>
            </Chapter>
        </Narrative>
    </Page>
)

//   publish
export default Overview

// end of file
