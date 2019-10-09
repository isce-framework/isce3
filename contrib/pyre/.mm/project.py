# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# california institute of technology
# (c) 1998-2019 all rights reserved
#


def requirements(package):
    """
    Build a dictionary with the external dependencies of the {pyre} project
    """

    # build the package instances
    packages = [
        package(name='cuda', optional=True),
        package(name='gsl', optional=True),
        package(name='libpq', optional=True),
        package(name='mpi', optional=True),
        package(name='python', optional=False),
        ]

    # build a dictionary and return it
    return { package.name: package for package in packages }


# end of file
