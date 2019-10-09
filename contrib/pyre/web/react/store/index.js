// -*- web -*-
//
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// externals
// store setup
import {
    createStore as createReduxStore,
    combineReducers, applyMiddleware, compose
} from 'redux'
// browser history
import createHistory from 'history/createBrowserHistory'
// react-router's connection of history to the store
import {
    ConnectedRouter,
    routerReducer as router,
    routerMiddleware
} from 'react-router-redux'

// alec's connector of viewport state to the store
import {
    responsiveStoreEnhancer, calculateResponsiveState
} from 'redux-responsive'

// locals
import browser from './browser'

// create a browser history
const history = createHistory()
// build the middleware for intercepting and dispatching navigation actions
const middleware = routerMiddleware(history)

// assemble my reducers
const reducer = combineReducers({
    browser,
    router,
})

// my store factory
const createStore = () => createReduxStore(
    reducer,
    compose(
        responsiveStoreEnhancer,
        applyMiddleware(middleware)
    )
)

// my store
const store = createStore()

// make sure we track the window as it changes size
window.addEventListener('resize', () =>
    // update the redux store
    store.dispatch(calculateResponsiveState(window))
)

// publish
export default store

// end of file
