// -*- web -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// externals
import React from 'react'
// locals
import styles from './styles'

// declare
const Footer = () => (
    <footer style={styles.footer}>
        <div style={styles.colophon}>
            <span style={styles.copyright}>
                copyright &copy; 1998-2019
                &nbsp;
                <a href="http://www.orthologue.com/michael.aivazis">
                    michael aïvázis
                </a>
                &nbsp;
                -- all rights reserved
            </span>
        </div>
    </footer>
)

// export
export default Footer

// end of file
