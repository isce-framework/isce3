# -*- coding: utf-8 -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#


# externals
import os


# platform hook
def platform(builder):
    """
    Decorate the builder with platform specific options
    """
    # get the platform id
    platform = builder.host.system
    # print('platform:', platform)

    # on darwin
    if platform == 'Darwin':
        # assume macports
        systemdir = '/opt/local'
        systemlibdir = os.path.join(systemdir, 'lib')
        systemincdir = os.path.join(systemdir, 'include')

        # set up python
        pythonVersion = '3.4'
        pythonMemoryModel = 'm'
        python = 'python' + pythonVersion
        pythonHome = os.path.join(
            systemdir, 'Library/Frameworks/Python.framework/Versions', pythonVersion)
        builder.requirements['python'].environ = {{
            'PYTHON': python,
            'PYTHON_PYCFLAGS': '-b',
            'PYTHON_DIR': systemdir,
            'PYTHON_LIBDIR': os.path.join(pythonHome, 'lib'),
            'PYTHON_INCDIR': os.path.join(pythonHome, 'include', python+pythonMemoryModel),
            }}

        # all done
        return builder

    # on linux
    if platform == 'Linux':
        # on normal distributions
        systemdir = '/usr'
        systemlibdir = os.path.join(systemdir, 'lib')
        systemincdir = os.path.join(systemdir, 'include')

        # set up python
        pythonVersion = '3.4'
        python = 'python' + pythonVersion
        builder.requirements['python'].environ = {{
            'PYTHON': python,
            'PYTHON_PYCFLAGS': '-b',
            'PYTHON_DIR': systemdir,
            'PYTHON_LIBDIR': os.path.join(systemdir, 'lib', python),
            'PYTHON_INCDIR': os.path.join(systemdir, 'include', python),
            }}

        # all done
        return builder

    # on all other platforms
    return builder


# end of file
