// -*- web -*-
//
// authors:
//   {project.authors}
//
// (c) {project.span} all rights reserved
//

// externals
import React from 'react'
import {{ connect }} from 'react-redux'
import {{ Link }} from 'react-router-dom'

// locals
import styles from './styles'

// declare
const header = ({{title}}) => (
    <header style={{styles.header}}>
        <img style={{styles.logo}} src="/graphics/logo.png" />
        <h1 style={{styles.title}}>{{title}}</h1>
        <nav style={{styles.nav}}>
            <Link to="/home" style={{{{...styles.navLink, ...styles.navLinkLast}}}}>home</Link>
        </nav>
    </header>
)

// grab the page title from the store
const getTitle = ({{navigation}}) => ({{title: navigation.get('title')}})

// publish
export default connect(getTitle)(header)

// end of file
