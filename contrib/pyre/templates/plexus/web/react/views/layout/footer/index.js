// -*- web -*-
//
// authors:
//   {project.authors}
//
// (c) {project.span} all rights reserved
//

// externals
import React from 'react'
import axios from 'axios'
// support
import Fetch from 'widgets/fetch'
// locals
import styles from './styles'

// helper
const restart = event => {{
    // stop the vent from bubbling up
    event.stopPropagation()
    // ask the server to restart
    axios.put('/action/meta/stop').then(() => null).catch(() => null)
    // close the window
    window.close()
}}

// declare and public
const footer = () => (
    <footer style={{styles.footer}}>
        <Fetch url="/query/meta/version">
            {{({{status, document: version}}) => status === "success" && (
                 <span style={{styles.serverVersion}}>
                     <span style={{styles.action}} onClick={{restart}}>{project.name}</span>:
                     version {{version}}
                 </span>
             )}}
        </Fetch>
        <span style={{styles.copyright}}>
            Copyright &copy; {project.span}
            &nbsp;<a href="http://{project.live.name}" >{project.affiliations}</a>&nbsp;
            -- all rights reserved
        </span>
    </footer>
)

// export
export default footer

// end of file
