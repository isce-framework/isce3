# -*- Python -*-
# -*- coding: utf-8 -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#


# import and publish pyre symbols
from pyre import (
    # protocols, components and traits
    schemata, constraints, properties, protocol, component, foundry,
    # decorators
    export, provides,
    # the manager of the pyre runtime
    executive,
    # miscellaneous
    tracking, units
    )


# bootstrap
package = executive.registerPackage(name='{project.name}', file=__file__)
# save the geography
home, prefix, defaults = package.layout()

# publish local modules
from . import (
    meta, # meta-data
    extensions, # my extension module
    )

# administrivia
def copyright():
    """
    Return the copyright note
    """
    # pull and print the meta-data
    return print(meta.header)


def license():
    """
    Print the license
    """
    # pull and print the meta-data
    return print(meta.license)


def built():
    """
    Return the build timestamp
    """
    # pull and return the meta-data
    return meta.date


def credits():
    """
    Print the acknowledgments
    """
    return print(meta.acknowledgments)


def version():
    """
    Return the version
    """
    # pull and return the meta-data
    return meta.version


# plexus support
from .components import plexus, action, command


# end of file
