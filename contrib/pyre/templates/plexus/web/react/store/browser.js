// -*- web -*-
//
// authors:
//   {project.authors}
//
// (c) {project.span} all rights reserved
//
//

// externals
import {{ createResponsiveStateReducer }} from 'redux-responsive'

// create a reducer that is aware of the height and width of the browser
const reducer = createResponsiveStateReducer(null, {{
    extraFields: () => ({{
        width: window.innerWidth,
        height: window.innerHeight,
    }})
}})

// publish
export default reducer

// end of file
