// -*- javascript -*-
//
// authors:
//   {project.authors}
//
// (c) {project.span} all rights reserved
//

// webpack imports
var webpack = require('webpack')
var path = require('path')
var HtmlWebpackPlugin = require('html-webpack-plugin');


var rootDir = path.join(__dirname, '..')
var sourceDir = path.join(rootDir, 'react')
var configDir = path.join(rootDir, 'config')
var buildDir = path.join(rootDir, 'build')

// export webpack configuration object
module.exports = {{
    entry: {{
        {project.name}: path.join(sourceDir, '{project.name}.js'),
    }},
    output: {{
        filename: path.join('/[name].js'),
        path: buildDir,
    }},
    module: {{
        loaders: [
            {{
                test: /\.jsx?$/,
                loader: 'babel',
                include: [
                    sourceDir,
                ],
                query: {{
                    extends: path.join(configDir, 'babelrc')
                }}
            }},
        ],
    }},
    resolve: {{
        extensions: ['', '.js', '.jsx'],
        root: [sourceDir],
    }},
    plugins: [
        new HtmlWebpackPlugin({{
            template: path.join(sourceDir, '{project.name}.html'),
            inject: 'body',
            filename: path.join(buildDir, '{project.name}.html')
        }}),
    ],

    devtool: "inline-source-map",
}}

// end of file
