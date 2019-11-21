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
import ReactDom from 'react-dom'
import {{ Provider }} from 'react-redux'
import 'babel-polyfill'

// locals
// my redux store
import store from './store'
// my root view
import Layout from './views'

// render
ReactDom.render((
    <Provider store={{store}}>
        <Layout/>
    </Provider>
), document.querySelector('#{project.name}'))

// end of file
