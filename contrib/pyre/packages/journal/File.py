# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# packages
import pyre


# the implemented interfaces
from .Device import Device


# declaration
class File(pyre.component, family="journal.devices.file", implements=Device):
    """
    This is a sample documentation string for class Console
    """


    # types
    from .TextRenderer import TextRenderer


    # public state
    log = pyre.properties.ostream()
    log.doc = "the file in which to save the journal entries"

    renderer = Device.Renderer(default=TextRenderer)
    renderer.doc = "the formatting strategy for journal entries"


    # interface
    @pyre.export
    def record(self, page, metadata):
        """
        Record a journal entry
        """
        # get the renderer to produce the text
        for line in self.renderer.render(page, metadata):
            # print it to stdout
            print(line, file=self.log)
        # flush the output
        self.log.flush()
        # and return
        return self


# end of file
