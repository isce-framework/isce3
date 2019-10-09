// -*- web -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// externals
import React from 'react'

// render
const SectionTitle = ({logo, children, style}) => (
    <div style={style.bar}>
        <h1 style={style.title}>
            {children}
        </h1>
        {logo && <img style={style.logo} src="/graphics/logo.png" />}
    </div>
)

// defaults
SectionTitle.defaultProps = {
    logo: false,
}

// publish
export default SectionTitle

// end of file
