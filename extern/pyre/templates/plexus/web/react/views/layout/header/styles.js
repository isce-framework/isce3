// -*- web -*-
//
// authors:
//   {project.authors}
//
// (c) {project.span} all rights reserved
//

// get the colors
import {{ wheel, semantics }} from 'palette'
// base styling
import base from '../styles'

// publish
export default {{
    // the style of the enclosing tag
    header: {{
        // for my parent
        flexGrow: "0",
        flexShrink: "0",

        // styling
        color: wheel.soapstone,
        backgroundColor: wheel.chalk,
        borderBottom: `1px solid ${{wheel.soapstone}}`,

        fontSize: "60%",
        padding: "1.0em",

        // for my children
        display: "flex",
        flexDirection: "row",
        alignItems: "center",
    }},

    // the logo
    logo: {{
        // for my parent
        flexGrow: "0",
        flexShrink: "0",

        // styling
        height: "2.0em",
        padding: "0.5em 1.0em",
    }},

    // the title
    title: {{
        // for my parent
        flexGrow: "1",
        flexShrink: "0",

        // styling
        fontSize: "150%",
        fontWeight: "normal",

        textAlign: "center",
        textTransform: "uppercase",
        color: semantics.title,

        margin: "0 auto",
        padding: "0.5em",
    }},

    // navigation
    nav: {{
        // for my parent
        flexGrow: "0",
        flexShrink: "0",

        // styling
        padding: "0.5em 1.0em 0.25em 1.0em",
        height: "2.0em",

        // for my children
        display: "flex",
        flexDirection: "row",
    }},

    navLink: {{
        whiteSpace: "nowrap",
        paddingLeft: "1.0em",
        paddingRight: "1.0em",
        lineHeight: "2.0em",
        color: semantics.link,
        textTransform: "lowercase",
        borderRight: `1px solid ${{wheel.soapstone}}`,
    }},

    navLinkLast: {{
        borderRight: "none",
        paddingRight: "0.0em",
    }},
}}

// end of file
