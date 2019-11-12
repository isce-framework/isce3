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
import {
    Content, Narrative, Page, Paragraph, Pyre,
    Section, Subsection, TableOfContents
} from 'widgets/doc'

// locals
import styles from './styles'

// dress up the section title as a link to the syllabus
const title = (
    <span>A short course on <Pyre/></span>
)

// declaration
const Syllabus = () => (
    <Page>
        <Narrative>
            <Section id="syllabus" style={styles} title={title}>
                <Subsection title="Syllabus">
                    <Paragraph>
                        The material here is derived from a semester long course on using
                        <Pyre/> to write scientific applications. The lecture notes are written
                        using <code>beamer</code> and are bundled with the source code.
                    </Paragraph>
                </Subsection>

                <TableOfContents>
                    <Content title="from the end user's point of view">
                        <Link to="/docs/course/overview">
                            overview of <Pyre/>
                        </Link>
                        <Content title="capturing user choices"/>
                        <Content title="writing components"/>
                    </Content>
                    <Content title="for application developers">
                        <Content title="assembling applications"/>
                        <Content title="command line user interfaces"/>
                        <Content title="writing extensions in C and C++"/>
                        <Content title="support: journal, memory, arrays, MPI, CUDA, databases"/>
                    </Content>
                    <Content title="framework internals">
                        <Content title="data structures, evaluation networks,
                                        traits and their schemata"/>
                        <Content title="component configuration"/>
                        <Content title="runtime environment discovery"/>
                        <Content title="advanced techniques: user interfaces,
                                        asynchronous and concurrent execution" />
                    </Content>
                </TableOfContents>
            </Section>
        </Narrative>
    </Page>
)

//   publish
export default Syllabus

// end of file
