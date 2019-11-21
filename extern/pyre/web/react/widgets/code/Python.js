// -*- web -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// externals
import React from 'react'
import Highlight from 'react-syntax-highlighter'

// locals
import styles from './styles'
// theming support
import theme from './hljs'

// dress up
const Python = ({children}) => (
    <Highlight language="python"
               style={theme}
               customStyle={styles.code}
               showLineNumbers="true"
               lineNumberContainerStyle={styles.lineNumberContainer}
               lineNumberStyle={styles.lineNumber}>
        {children}
    </Highlight>
)

// and publish
export default Python

// end of file
