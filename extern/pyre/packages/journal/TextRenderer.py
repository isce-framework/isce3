# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# packages
import pyre


# the implemented interfaces
from .Renderer import Renderer


# declaration
class TextRenderer(pyre.component, family="journal.renderers.plain", implements=Renderer):
    """
    A plain text renderer
    """


    # public state
    header = pyre.properties.str(default=">>")
    header.doc = "the marker to use while rendering the diagnostic metadata"

    body = pyre.properties.str(default="--")
    body.doc = "the marker to use while rendering the diagnostic body"


    # interface
    @pyre.provides
    def render(self, page, metadata):
        """
        Convert the diagnostic information into a form that a device can record
        """
        # build the header
        # the marker
        header = [" {} ".format(self.header)]
        # the filename
        try:
            header.append("{filename}:".format(**metadata))
        except KeyError:
            pass
        # the line number
        try:
            header.append("{line}: ".format(**metadata))
        except KeyError:
            pass
        # the severity
        try:
            header.append("{severity}".format(**metadata))
        except KeyError:
            pass
        # the channel
        try:
            header.append("({channel}):".format(**metadata))
        except KeyError:
            pass
        # the function
        try:
            header.append(" in {function!r}:".format(**metadata))
        except KeyError:
            pass

        # return the header
        yield "".join(header)
        # and the body
        for line in page:
            yield " {} {} ".format(self.body, line)

        # all done
        return


# end of file
