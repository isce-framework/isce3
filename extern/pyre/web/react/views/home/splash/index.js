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

// declaration
const Splash = () => (
    <div style={styles.section}>
        <h3 style={styles.leader}>an architecture for</h3>
        <h1 style={styles.title}>scientific applications</h1>
        <p style={styles.abstract}>
            pyre is an open source application framework for building
            scalable, user-friendly scientific applications
        </p>

        <nav style={styles.navigation}>
            <div style={styles.button}>
                <Link to="/install" style={styles.label}>install</Link>
            </div>
            <div style={styles.button}>
                <Link to="/docs" style={styles.label}>documentation</Link>
            </div>
        </nav>

    </div>
)

//   publish
export default Splash

// end of file
