// -*- web -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// colors
import { wheel, semantic } from 'palette'


// styling common to all content levels
const structure = {

    container: {
        display: "flex",
        flexDirection: "column",
        overflowY: "auto",
        margin: "1em auto",
    },

    bar: {
        display: "flex",
        flexDirection: "row",
        alignItems: "center",
        margin: "1.0em 0.0em 1.0em 0.0em",
    },

    title: {
        textTransform: "uppercase",
        color: semantic.section.title.text,
    },

    logo: {
        height: "1.25em",
        marginLeft: "auto",
    },
}


// define
const document = {

    pyre: {
        color: semantic.pyre,
    },

    page: {
        display: "flex",
        flexDirection: "row",
        margin: "0em auto",
        justifyContent: "center",
        alignItems: "top",
    },

    sidebar: {
        margin: "0em 1em 0em 0em"
    },

    narrative: {
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        fontSize: "150%",
        lineHeight: "150%",
        margin: "0em 2em",
    },

    body: {
        textAlign: "justify",
        margin: "0.75em 0.0em",
        padding: "0.0em",
    },

    chapter: {

        container: {
            ...structure.container,
        },

        bar: {
            ...structure.bar,
            padding: "0.25em 1.5em",
            backgroundColor: semantic.section.title.banner,
        },

        title: {
            ...structure.title,
            fontWeight: "normal",
            fontSize: "180%",
            lineHeight: "200%",
        },

        logo: {
            ...structure.logo,
        },
    },

    section: {

        container: {
            ...structure.container,
        },

        bar: {
            ...structure.bar,
            padding: "0.25em 1.5em",
            backgroundColor: semantic.section.title.banner,
        },

        title: {
            ...structure.title,
            fontWeight: "normal",
            fontSize: "150%",
            lineHeight: "200%",
        },

        logo: {
            ...structure.logo,
        },
    },

    subsection: {
        container: {
            ...structure.container,
        },

        bar: {
            ...structure.bar,
        },

        title: {
            ...structure.title,
            fontWeight: "normal",
            fontSize: "120%",
            lineHeight: "150%",
        },
    },

    toc: {
        container: {
            ...structure.container,
            flexGrow: 0,
            margin: "1.0em 0.0em",
        },

        title: {
            ...structure.title,
            fontSize: "120%",
            lineHeight: "150%",
        },

        item: {
            lineHeight: "175%",
        },

        contents: {
            marginLeft: "2em",
        }
    },
}

// publish
export default document



// end of file
