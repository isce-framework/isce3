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
const Section = ({id, title, logo, children, style}) => {
    // mix the styles
    const mixed = {
        container: {
            ...document.section.container,
            ...style.container,
        },

        bar: {
            ...document.section.bar,
            ...style.bar,
        },

        title: {
            ...document.section.title,
            ...style.title,
        },

        logo: {
            ...document.section.logo,
            ...style.logo,
        },
    }
    // render
    return (
        <section id={id} style={mixed.container}>
            <Title logo={logo} style={mixed}>
                {title}
            </Title>
            {children}
        </section>
    )
}

// defaults
Section.defaultProps = {
    id: "unused",
    logo: true,
    title: "please specify the section title",
}

// publish
export default Section

// end of file
