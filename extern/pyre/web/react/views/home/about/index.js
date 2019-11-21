// -*- web -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// externals
import React from 'react'

// support
import { Narrative, Page, Pyre, Section, Paragraph } from 'widgets/doc'
// locals
import styles from './styles'

import Ball from './ball'
import Shape from './shape'
import Gauss from './gauss'
import Launch from './launch'
import Launch3D from './launch3d'


// declaration
const About = () => (
    <Page>
        <Narrative>
            <Section id="about" style={styles} title="quick introduction">

                <Paragraph>
                    <Pyre/> is an open source application framework written in <a
                    href="http://www.python.org">python</a>. It's an attempt to bring state of
                    the art software design practices to scientific computing. The goal is to
                    provide a strong skeleton on which to build scientific codes by steering
                    the implementation towards usability and maintainability.
                </Paragraph>

                <Paragraph>
                    The basic conceptual building block in <Pyre/> is the <em>component</em>.
                    Components are classes that specifically grant access
                    to some of their state to the application end user. Component authors
                    provide default values to be used in case the user doesn't make a choice,
                    but the user is explicitly given complete control and can override them
                    during component configuration.
                </Paragraph>

                <Paragraph>
                    Here is a simple component that represents a multi-dimensional ball. It has
                    two user-configurable properties, and a single method that computes its
                    measure:
                </Paragraph>

                <Ball/>

                <Paragraph>
                    Packages that support more than one kind of shape should probably declare
                    a <em>protocol</em> to capture their basic properties and behaviors, much
                    like abstract base classes do for their sub-classes.
                </Paragraph>

                <Shape/>

                <Paragraph>
                    Then, applications can use the protocol to express
                    a <em>requirement</em> for some kind of shape, and defer to the user
                    for the actual choice. If the user does not express an opinion, the
                    default specified by Shape will be used.
                </Paragraph>

                <Gauss/>

                <Paragraph>
                    Assuming the application above is in a file <code>gauss.py</code> that has
                    execute permissions, you can launch the app from the command line to
                    compute the volume of the default shape, which is a unit circle with its
                    center at the origin:
                </Paragraph>

                <Launch/>

                <Paragraph>
                    Our implementation is dimension independent, and gets its dimensionality
                    clues from the number of coördinates it takes to specify the center of the
                    ball. We can compute the volume of the unit sphere in three dimensions by
                    invoking the app with the following command line arguments:
                </Paragraph>

                <Launch3D/>

                <Paragraph>
                    You play with the radius by providing values for 
                    the <code>--shape.radius</code> command line argument.  For more details, and
                    more sophisticated examples, please take a look at the tutorials.
                </Paragraph>

                <Paragraph>
                    Frequently used configurations can be stored in files in a variety of
                    formats and loaded both automatically and on demand. For more sophisticated
                    applications, <Pyre/> provides infrastructure for storing configuration in
                    databases.
                </Paragraph>

            </Section>
        </Narrative>
    </Page>
)

//   publish
export default About

// end of file
