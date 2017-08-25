#
# Author: Joshua Cohen
# Copyright 2017
#

import os
from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize

source_dir = ".."

obj_files = ['Ellipsoid',
             'Interpolator',
             'LinAlg',
             'Orbit',
             'Peg',
             'Pegtrans',
             'Poly1d',
             'Poly2d',
             'Position']

source_files = [os.path.join(source_dir,f+'.cpp') for f in obj_files]

setup(ext_modules = cythonize(Extension(
    "iscecore",
    sources=source_files+['iscecore.pyx'],
    include_dirs=['../../../'],
    extra_compile_args=['-std=c++11','-fPIC'],
    extra_link_args=['-lm'],
    language="c++"
    ))
)
