// -*- web -*-
//
// authors:
//   {project.authors}
//
// (c) {project.span} all rights reserved
//

// externals
import {{ createStore as createReduxStore, applyMiddleware, combineReducers, compose }} from 'redux'
import thunk from 'redux-thunk'
import {{ responsiveStoreEnhancer, calculateResponsiveState }} from 'redux-responsive'

// locals
import browser from './browser'
import navigation from './navigation'

// assemble my reducers
const reducer = combineReducers({{
    browser,
    navigation,
}})

// my store factory
const createStore = () => createReduxStore(
    reducer,
    compose(
        responsiveStoreEnhancer,
        applyMiddleware(thunk),
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
