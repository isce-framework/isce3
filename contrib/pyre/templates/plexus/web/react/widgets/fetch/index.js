// -*- web -*-
//
//
// authors:
//   {project.authors}
//
// (c) {project.span} all rights reserved
//

// externals
import React from 'react'
import PropTypes from 'prop-types'
import axios from 'axios'

// asynchronous access to server assets
class Fetch extends React.Component {{

    // render
    render() {{
        // render my children
        return this.props.children(this.state)
    }}

    // meta-methods
    constructor(...args) {{
        // chain up
        super(...args)
        // bind my methods
        this.fetch = this.fetch.bind(this)
        // initialize my state
        this.state = {{
            status: null,
            document: null,
        }}
        // all done
        return
    }}

    // hooks
    componentDidMount() {{
        // get the document from the server
        this.fetch()
        // all done
        return
    }}

    // implementation  details
    async fetch() {{
        // make storage for the target document
        let document
        // attempt to
        try {{
            // fetch the document
            document = (await axios.get(this.props.url)).data
        // and if it fails
        }} catch (error) {{
            // update
            this.setState({{
                // the document status
                status: "failed",
                // and nullify the document
                document: null,
            }})
            // and bail
            return
        }}
        // otherwise, update my state
        this.setState({{
            // attach the document
            document,
            // and indicate success
            status: "success"
        }})
        // all done
        return
    }}

    // data
    state = {{
        document: null,
        status: "pending",
    }}

    // prop typing
    static propTypes = {{
        url: PropTypes.string.isRequired,
        children: PropTypes.func.isRequired,
    }}
}}

// publish
export default Fetch

// end of file
