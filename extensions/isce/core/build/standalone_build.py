#
# Author: Joshua Cohen
# Copyright 2017
#

import os
from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize

source_dir = "src"
include_dir = "include"
pyx_dir = "cython"

obj_files = ['Ellipsoid',
             'Interpolator',
             'LinAlg',
             'Orbit',
             'Peg',
             'Pegtrans',
             'Poly1d',
             'Poly2d',
             'Position'
            ]

header_files = [os.path.join(include_dir,f+'.h') for f in obj_files]
header_files += os.path.join(include_dir,'isceLibConstants.h')
source_files = [os.path.join(source_dir,f+'.cpp') for f in obj_files]

setup(ext_modules = cythonize(Extension(
    "isceLib",
    sources=source_files+[os.path.join('cython','isceLib.pyx')],
    include_dirs=[include_dir],
    extra_compile_args=['-std=c++11','-fPIC'],
    extra_link_args=['-lm'],
    language="c++"
    ))
)
