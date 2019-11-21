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
const Chapter = ({id, title, logo, children, style}) => {
    // mix the styles
    const mixed = {
        container: {
            ...document.chapter.container,
            ...style.container,
        },

        bar: {
            ...document.chapter.bar,
            ...style.bar,
        },

        title: {
            ...document.chapter.title,
            ...style.title,
        },

        logo: {
            ...document.chapter.logo,
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
Chapter.defaultProps = {
    id: "unused",
    logo: true,
    title: "please specify the chapter title",
}

// publish
export default Chapter

// end of file
