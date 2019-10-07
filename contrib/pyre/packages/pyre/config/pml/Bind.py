# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


import collections
from .Node import Node


class Bind(Node):
    """
    Handler for the bind tag in pml documents
    """

    # constants
    elements = ()


    # interface
    def content(self, text, locator):
        """
        Store the value of the key
        """
        text = text.strip()
        if text:
            self.text.append(text)
        return


    def notify(self, parent, locator):
        """
        Let {parent} now that processing this bind tag is complete
        """
        # convert the collected text into a value
        # flatten a single line of text into a string
        if len(self.text) == 1:
            value = self.text[0]
        # pass multi-line input as a list
        else:
            value = self.text
        # make an assignment event
        event = self.Assignment(
            key=self.key, value=value, locator=self.newLocator(locator))
        # and pass it on to my parent
        parent.assignment(event)
        # all done
        return


    # meta methods
    def __init__(self, parent, attributes, locator):
        self.key = attributes['property'].split(self.separator)
        self.text = []
        return


# end of file
