# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# the framework
import pyre
# my superclass
from .ProjectTemplate import ProjectTemplate


# declaration
class Django(ProjectTemplate, family='pyre.weaver.projects.django'):
    """
    Encapsulation of the project information
    """


    # additional user configurable state
    template = pyre.properties.str(default='django')
    template.doc = "the project template"


# end of file
