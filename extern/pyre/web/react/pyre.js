// -*- web -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// externals
import React from 'react'
import ReactDom from 'react-dom'
import { Provider } from 'react-redux'
import 'babel-polyfill'

// locals
// my redux store
import store from './store'
// my root view
import Layout from './views'

// render
ReactDom.render((
    <Provider store={store}>
        <Layout/>
    </Provider>
), document.querySelector('#pyre'))

// end of file
