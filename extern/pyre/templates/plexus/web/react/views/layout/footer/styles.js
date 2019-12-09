// -*- web -*-
//
// authors:
//   {project.authors}
//
// (c) {project.span} all rights reserved
//

// get the colors
import {{ wheel }} from 'palette'

// publish
export default {{
    //
    footer: {{
        // for my parent
        flexGrow: "0",
        flexShrink: "0",

        // for my children
        color: wheel.soapstone,
        backgroundColor: wheel.chalk,
        borderTop: `1px solid ${{wheel.soapstone}}`,

        fontSize: "50%",
        lineHeight: "150%",
        padding: "1.0em",
        // boxShadow: "0 -3px 10px 0 rgba(0, 0, 0, .0785)",

        display: "flex",
        flexDirection: "row",
    }},

    copyright: {{
        // for my parent
        flexGrow: "1",
        flexShrink: "0",

        // for my children
        fontWeight: "normal",
        textAlign: "right",
    }},

    social: {{
        marginLeft: "1.5em",
    }},

    action: {{
        cursor: "pointer",
    }},

}}

// end of file
