// -*- web -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

const wheel = {
    // greys
    "obsidian": "#000",
    "basalt": "#333",
    "steel": "#666",
    "aluminum": "#a5a5a5",
    "soapstone": "#dadada",
    "cement": "#eee",
    "flour": "#fafafa",
    "milk": "#fdfdfd",
    "chalk": "#fff",
    }

const semantic = {
    pyre: "#aab141",

    error: "#ffa0a0",

    title: "#ff973e",

    section: {
        title: {
            "text": "#7a7864",
            "banner": "#f4f2e4",
        },
    },

    button: {
        "border": "#bbb",
        "label": "#4daae8",
    },
}

// theme for syntax highlighting code snippets
const theme = {
    background: "#fdfdfd",
    background: "#fffef2",
    normal: "#5e6e5e",

    pyre: semantic.pyre,

    comment: "#ccc",
    string: "#999",

    type: "#0080c0",
    literal: "#0080c0",
    number: "#0080c0",
    builtin: "#0080c0",
    parameters: "#0080c0",

    keyword: "#ad2bee",
    selector: "#ad2bee",

    meta: "#20b2aa", // python decorators, ...
    lineNumber: "#ddd",

    // unmapped
    quote: "#687d68",
}

// publish
export { wheel, semantic, theme }

// end of file
