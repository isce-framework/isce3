// -*- web -*-
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// externals
var path = require('path')

// the project root
var root = path.join(__dirname, '..')
// derived paths
var build = path.join(root, 'build')
var source = path.join(root, 'react')
var config = path.join(root, 'config')

// publish
module.exports = {
    // paths
    root: root,
    source: source,
    config: config,
    build: build,
    // entry points
    template: path.join(source, 'pyre.html'),
    target: path.join(build, 'pyre.html'),
    client: path.join(source, 'pyre.js'),
    // configuration files
    babel: path.join(config, 'babelrc'),
    webpack: path.join(config, 'webpack.js'),
}

// end of file
