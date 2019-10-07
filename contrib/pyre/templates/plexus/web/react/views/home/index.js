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
import {{ connect }} from 'react-redux'

// locals
import styles from './styles'

// declaration
const Home = ({{setTitle}}) => {{
    // set the page title
    setTitle("{project.name}");

    // render
    return (
        <main style={{styles.main}}>
        </main>
    )
}}

// store access
const store = null

// actions
import {{ setPageTitle }} from 'actions/navigation'

// action dispatch
const actions = (dispatch) => ({{
    setTitle: title => dispatch(setPageTitle(title)),
}})

// publish
export default connect(store, actions)(Home)

// end of file
