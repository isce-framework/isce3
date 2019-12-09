// -*- web -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// externals
import React from 'react'
import { Switch, Route } from 'react-router-dom'

// locals
import Cover from './Cover'
import Course from './course'

// declaration
const Documentation = () => (
    <Switch>
        <Route exact path="/docs" component={Cover}/>
        <Route path="/docs/course" component={Course}/>
    </Switch>
)

//   publish
export default Documentation

// end of file
