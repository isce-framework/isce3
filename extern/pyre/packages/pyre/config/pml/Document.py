# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


from pyre.xml.Document import Document as Base


# import the handlers
from .Bind import Bind
from .Component import Component
from .Configuration import Configuration
from .Package import Package


class Document(Base):
    """
    The anchor point for the handlers of the pml document tags
    """

    # constants
    root = "config" # the top level element tag

    # get access to the element descriptor factory
    import pyre.xml
    # the element descriptors
    bind = pyre.xml.element(tag="bind", handler=Bind)
    component = pyre.xml.element(tag="component", handler=Component)
    config = pyre.xml.element(tag="config", handler=Configuration)
    package = pyre.xml.element(tag="package", handler=Package)


    # interface
    def onConfiguration(self, node):
        """
        Handle the top level tag
        """
        # {node} is the rep of the <config> tag, which stores the configuration events in a
        # {pyre.config.Configuration} instance; grab the configuration instance and make it my
        # contents (perhaps I should just grab the event iterable instead)
        self.dom = node.configuration
        # all done
        return


# end of file
