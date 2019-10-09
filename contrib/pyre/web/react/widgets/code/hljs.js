// -*- web -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// theming support
import { theme } from 'palette'

// publish
export default {
    hljs: {
        display: "block",
        overflowX: "auto",
        padding: "0.5em",

        color: theme.normal,
        background: theme.background,
        borderLeft: `3px solid ${theme.pyre}`,
    },

    "hljs-emphasis": {
        fontStyle: "italic",
    },

    "hljs-strong": {
        fontWeight: "bold",
    },

    "hljs-comment": {
        fontStyle: "italic",
        color: theme.comment,
    },
    "hljs-variable": {
        color: "#e6193c",
    },
    "hljs-template-variable": {
        color: "#e6193c",
    },
    "hljs-attribute": {
        color: "#e6193c",
    },
    "hljs-tag": {
        color: "#e6193c",
    },
    "hljs-name": {
        color: "#e6193c",
    },
    "hljs-regexp": {
        color: "#e6193c",
    },
    "hljs-link": {
        color: "#e6193c",
    },
    "hljs-number": {
        color: theme.number,
    },
    "hljs-meta": {
        color: theme.meta,
    },
    "hljs-built_in": {
        color: theme.builtin,
    },
    "hljs-builtin-name": {
        color: theme.builtin,
    },
    "hljs-literal": {
        color: theme.literal,
    },
    "hljs-type": {
        color: theme.type,
    },
    "hljs-params": {
        color: theme.parameters,
    },
    "hljs-string": {
        fontStyle: "italic",
        color: theme.string,
    },
    "hljs-symbol": {
        color: "#29a329",
    },
    "hljs-bullet": {
        color: "#29a329",
    },
    "hljs-title": {
        color: "#3d62f5",
    },
    "hljs-section": {
        color: "#3d62f5",
    },
    "hljs-keyword": {
        color: theme.keyword,
    },

    "hljs-selector-tag": {
        color: theme.selector,
    },
    "hljs-selector-id": {
        color: "#e6193c",
    },
    "hljs-selector-class": {
        color: "#e6193c",
    },

    // unmapped
    "hljs-quote": {
        color: theme.quote,
    },
};

// end of file
