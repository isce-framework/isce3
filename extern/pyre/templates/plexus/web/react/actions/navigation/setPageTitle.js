// -*- web -*-
//
//
// authors:
//   {project.authors}
//
// (c) {project.span} all rights reserved
//

// locals
import {{ NAVIGATION_PAGE_TITLE }} from './types'

// reducer
export default (title) => ({{
    type: NAVIGATION_PAGE_TITLE,
    payload: {{title}},
}})

// end of file
