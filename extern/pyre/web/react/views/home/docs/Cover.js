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
import { Narrative, Page, Pyre, Section, Subsection, Paragraph } from 'widgets/doc'
// locals
import styles from './styles'

// declaration
const Documentation = () => (
    <Page>
        <Narrative>
            <Section id="docs" style={styles} title="Documentation">
                <Paragraph>
                    This section is under active development. New material shows up all the
                    time. Many of the links do not work yet because the material is being
                    developed. I use it as my to-do list, so bear with me.
                </Paragraph>

                <Paragraph>
                    <Pyre/> is a tool for application developers. The intended audience are
                    domain experts with strong programming skills that are looking to make
                    their work accessible to experts and non-experts alike.  With this in mind,
                    the documentation in this section uses second person pronouns to refer to
                    you, the application developer. You know who you are. We refer to your
                    intended audience as <em>end users</em>, or simply the <em>users</em>.
                </Paragraph>

                <Paragraph>
                    The goal of <Pyre/> is to provide sensible, orthogonal mechanisms for
                    expressing the user facing part of your code so you can focus on the
                    science.  When designing a user experience, be it the name of your
                    application, the layout of your configuration files, or a graphical
                    interface, it is important to see things from their perspective. You should
                    build user stereotypes with differet skill sets and attitudes towards your
                    code, and make sure you understand who and why is included as the target
                    audience of a feature.
                </Paragraph>

                <Paragraph>
                    The current plan is to have three guides.  The first will be suitable for
                    your end users to get to know <Pyre/> and the way it shapes their
                    experience with your software. The second is a guide that describes what
                    <Pyre/> can do for you. The last one is about the implementation of the
                    framework, its algorithms and data structures, and is suitable if you are
                    interested in <Pyre/> internals.
                </Paragraph>

                <Paragraph>
                    Another resource is a web-friendly version of the <Link
                    to="docs/course">lecture notes</Link> for a semester long course on
                    scientific computing using <Pyre/>. The course slides are produced using
                    <code>beamer</code>, and they are bundled with the source code.
                </Paragraph>
            </Section>
        </Narrative>
    </Page>
)

//   publish
export default Documentation

// end of file
