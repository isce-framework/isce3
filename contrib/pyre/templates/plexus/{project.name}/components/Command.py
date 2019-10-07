# -*- coding: utf-8 -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#


# access the pyre framework
import pyre
# my protocol
from .Action import Action as action


# class declaration
class Command(pyre.panel(), implements=action):
    """
    Base class for {{{project.name}}} commands
    """


# end of file
