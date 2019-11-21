// -*- web -*-
//
//
// authors:
//   {project.authors}
//
// (c) {project.span} all rights reserved
//

// externals
import immutable from 'immutable'

// support
import {{
    NAVIGATION_PAGE_TITLE
}} from 'actions/navigation/types'

// initial state
const empty = immutable.Map({{
    title: '',
}})

// navigation support reducer
export default (navigation = empty, {{type, payload}}) => {{
    // set the page title
    if (type == NAVIGATION_PAGE_TITLE) {{
        // unpack
        const {{title}} = payload
        // update the page title
        navigation = navigation.set('title', title)
        // all done
        return navigation
    }}

    // if we get this far, we don't recognize the action; do nothing to the store
    return navigation
}}

// end of file
