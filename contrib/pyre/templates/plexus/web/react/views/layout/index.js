// -*- web -*-
//
// authors:
//   {project.authors}
//
// (c) {project.span} all rights reserved
//

// externals
import React from 'react'
import {{ BrowserRouter as Router, Switch, Route }} from 'react-router-dom'

// locals
import styles from './styles'
import Header from './header'
import Footer from './footer'
import Home from '../home'

// declare
const Layout = () => (
    <Router>
        <div style={{styles.layout}}>
            <Header/>
            <Switch>
                {{/* default landing spot */}}
                <Route path="/" component={{Home}}/>
            </Switch>
            <Footer/>
        </div>
    </Router>
)

// publish
export default Layout

// end of file
