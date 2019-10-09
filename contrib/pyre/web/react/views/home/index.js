// -*- web -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// externals
import React from 'react'
import { Switch, Route } from 'react-router-dom'

// support
import { Narrative, Page } from 'widgets/doc'

// locals
import styles from './styles'
import Splash from './splash'
import About from './about'
import Install from './install'
import Documentation from './docs'


// assemble the default view
const Default = () => (
    <div>
        <Splash/>
        <About/>
    </div>
)

// declaration
const Home = () => (
    <main style={styles.main}>
        <Switch>
            <Route exact path="/" component={Default}/>
            <Route path="/about" component={About}/>
            <Route path="/install" component={Install}/>
            <Route path="/docs" component={Documentation}/>
            <Route component={Default}/>
        </Switch>
    </main>
)

//   publish
export default Home

// end of file
