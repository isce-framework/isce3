# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import re
# the framework
import pyre
# my superclass
from .ProjectTemplate import ProjectTemplate


# declaration
class Plexus(ProjectTemplate, family='pyre.weaver.projects.plexus'):
    """
    Encapsulation of the project information
    """


    # additional user configurable state
    template = pyre.properties.str(default='plexus')
    template.doc = "the project template"


    # interface
    def blacklisted(self, filename):
        """
        Check whether {filename} is on the list of files to not expand
        """
        # check with my pile
        return self.blacklist.match(filename)


    # constants
    blacklist = re.compile("|".join([
        'jquery.js',
        '.+\.png',
        ]))


# end of file
