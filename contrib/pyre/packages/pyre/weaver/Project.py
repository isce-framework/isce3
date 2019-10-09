# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# the framework
import pyre
# my traits
from .Installation import Installation


# declaration
class Project(pyre.protocol, family='pyre.weaver.projects'):
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

    live = Installation()
    live.doc = "information about the remote host"


    # interface
    def blacklisted(self, filename):
        """
        Check whether {filename} is on the list of files to not expand
        """
        # nothing is blacklisted, by default
        return False


    # framework obligations
    @classmethod
    def pyre_default(cls, **kwds):
        """
        Build the preferred host implementation
        """
        # the default project is a plexus
        from .Plexus import Plexus
        return Plexus


# end of file
