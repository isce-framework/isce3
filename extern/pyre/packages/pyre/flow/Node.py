# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# support
import pyre
# my meta-class
from .FlowMaster import FlowMaster


# declaration
class Node(pyre.component, metaclass=FlowMaster, internal=True):
    """
    Base class for entities that participate in workflows
    """


    # public data
    # the object that watches over my traits
    pyre_status = None


    # public data
    @property
    def pyre_stale(self):
        """
        Retrieve my status
        """
        # delegate to my status manager
        return self.pyre_status.stale

    @pyre_stale.setter
    def pyre_stale(self, value):
        """
        Set my status
        """
        # delegate to my status manager
        self.pyre_status.stale = value
        # all done
        return


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # build my status tracker
        self.pyre_status = self.pyre_newStatus(node=self)
        # all done
        return


    # persistence
    def pyre_save(self):
        """
        Save my entire graph
        """
        # make a weaver
        weaver = pyre.weaver.weaver()
        # set the encoding
        weaver.language = self.encoding
        # open a file
        with open(f"{self.pyre_name}.{self.encoding}", mode='w') as stream:
            # assemble the document
            document = self.pyre_renderTraitValues(renderer=weaver.language)
            # get the weaver to do its things
            for line in weaver.weave(document=document):
                # place each line in the file
                print(line, file=stream)
        # all done
        return


    # flow hooks
    def pyre_newStatus(self, **kwds):
        """
        Build a handler for my status changes
        """
        # the handler is differentiated based on the type of flow node
        raise NotImplementedError(f"class '{type(self).__name__}' must override 'pyre_newStatus'")


    # debugging support
    def pyre_dump(self, channel, indent=' '*2, level=0):
        """
        Display information about me
        """
        # compute the margin
        margin = indent * level
        # make a weaver
        weaver = pyre.weaver.weaver()
        # pick the language
        weaver.language = self.encoding

        # assemble the document
        report = self.pyre_renderTraitValues(renderer=weaver.language)
        # go through each line
        for line in weaver.weave(document=report):
            # and inject into the channel
            channel.line(f"{margin}{line}")

        # all done
        return self


    # constants
    encoding = 'pfg'


# end of file
