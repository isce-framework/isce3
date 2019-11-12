# -*- coding: utf-8 -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#


def requirements(package):
    """
    Build a dictionary with the external dependencies of the {{{project.name}}} project
    """
    # build the package instances
    packages = [
        package(name='python', optional=False),
        ]
    # build a dictionary and return it
    return {{ package.name: package for package in packages }}


# end of file
