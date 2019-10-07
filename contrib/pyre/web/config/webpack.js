// -*- javascript -*-
// michael a.g. aïvázis
// orthologue
// (c) 1998-2019 all rights reserved
//

// get webpack
var webpack = require('webpack')
// access to environment variables
var process = require('process')
// path arithmetic
var path = require('path')

// the project layout
var layout = require('./layout')
// identify the build mode
var production = (process.env.NODE_ENV === 'production')
// do we need sane stack traces?
var devtool = production ? '' : 'inline-source-map'

// plugins
// we need at least this one
var HtmlWebpackPlugin = require('html-webpack-plugin');
// so use it initialize our pile
var plugins = [
    new HtmlWebpackPlugin({
        inject: 'body',
        template: layout.template,
        filename: layout.target
    })
]

// if we are building for production
if (production) {
    // use these additional production plugins
    plugins.push(
        new webpack.DefinePlugin({
            'process.env': {
                NODE_ENV: JSON.stringify('production')
            }
        }),
        new webpack.optimize.UglifyJsPlugin(),
        new webpack.optimize.OccurenceOrderPlugin(),
        new webpack.optimize.DedupePlugin()
    )
}

// export webpack configuration object
module.exports = {
    entry: {
        pyre: layout.client,
    },
    output: {
        filename: 'pyre.js',
        path: layout.build,
    },
    module: {
        loaders: [
            {
                test: /\.jsx?$/,
                loader: 'babel',
                include: [ layout.source, ],
                query: { extends: layout.babel }
            }, {
                test: /\.css$/,
                loaders: ['style', 'css'],
            }, {
                test: /\.(png|jpg|ttf|otf)$/,
                loader: 'url',
                query: {limit: 10*1024*1024},
            },
        ],
    },
    resolve: {
        extensions: ['', '.js', '.jsx'],
        root: [layout.source],
    },

    plugins: plugins,
    devtool: devtool,
}

// end of file
