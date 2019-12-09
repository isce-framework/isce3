// -*- web -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// externals
import React from 'react'
import { BrowserRouter as Router } from 'react-router-dom'

// locals
import styles from './styles'
import Header from './header'
import Footer from './footer'
import Home from '../home'

// declare
const Layout = () => (
    <Router>
        <div style={styles.layout}>
            <Header/>
            <Home/>
            <Footer/>
        </div>
    </Router>
)

// publish
export default Layout

// end of file
