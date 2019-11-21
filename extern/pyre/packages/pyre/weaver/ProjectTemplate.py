# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# the framework
import pyre
# my protocol
from .Project import Project
# my traits
from .Installation import Installation


# declaration
class ProjectTemplate(pyre.component, implements=Project):
    """
    Encapsulation of the project information
    """


    # user configurable state
    name = pyre.properties.str(default='project')
    name.doc = "the name of the project"

    authors = pyre.properties.str(default='[ replace with the list of authors ]')
    authors.doc = "the list of project authors"

    affiliations = pyre.properties.str(default='[ replace with the author affiliations ]')
    affiliations.doc = "the author affiliations"

    span = pyre.properties.str(default='[ replace with the project duration ]')
    span.doc = "the project duration for the copyright message"

    template = pyre.properties.str(default=None)
    template.doc = "the project template"

    live = Installation()
    live.doc = "information about the remote host"


# end of file
