# -*- coding: utf-8 -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#


# the project requirements hook
def requirements(package):
    """
    Build a dictionary with the external dependencies of the {{{project.name}}} project
    """

    # build the package instances; trim this list if your project doesn't need all of them
    packages = (
        package(name='python', optional=False),
        )
    # build a dictionary and return
    return {{ package.name: package for package in packages }}


# end of file 
