// -*- web -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// externals
import React from 'react'
import { Link } from 'react-router-dom'

// locals
import styles from './styles'

// table of links
const links = [
    { target: "/",        label: <img style={styles.logo} src="graphics/logo.png" />, },
    { target: "/about",   label: "about", },
    { target: "/install", label: "get", },
    { target: "/docs",    label: "documentation", },
    { target: "/contact", label: "contact", },
]

// declare
const header = () => (
    <header style={styles.header}>
        <div style={styles.bar}>
            <nav style={styles.nav}>
                {links.map( link => (
                     <Link key={link.target} to={link.target} style={styles.link}>
                         {link.label}
                     </Link>
                 ))}
            </nav>
        </div>
    </header>
)

// export
export default header

// end of file
