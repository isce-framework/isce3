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
import Title from './SectionTitle'

// render
const Subsection = ({id, title, logo, children, style}) => (
    <section id={id} style={style}>
        <Title logo={logo} style={document.subsection}>
            {title}
        </Title>
        {children}
    </section>
)

// defaults
Subsection.defaultProps = {
    id: "unused",
    logo: false,
    title: "please specify the subsection title",
}

// publish
export default Subsection

// end of file
