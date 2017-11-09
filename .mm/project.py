# -*- Python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# (c) 2003-2016 all rights reserved
#

def requirements(package):
    """
    Build a dictionary with the external dependencies of the {isce} project
    """
    # build the package instances
    packages = [
        package(name='python', optional=False)
        ]

    # turn into a dictionary and return it
    return { package.name: package for package in packages }

# end of file
