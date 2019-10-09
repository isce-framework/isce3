# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .Event import Event


# the class that captures key-value bindings
class Assignment(Event):
    """
    The resting place for key-value bindings
    """


    # public data
    name = None
    value = None
    locator = None


    # handlers
    def bind(self, token):
        """
        Extract my value from {token}
        """
        # easy enough
        self.value = '\n'.join(token.lexeme)
        # all done
        return


    def notify(self, parent):
        """
        Tell my {parent} there is an assignment waiting
        """
        # create a configuration event
        event = self.events.Assignment(
            key=self.name, value=self.value, locator=self.locator)
        # hand it to my {parent} for further processing
        return parent.assignment(event=event)


    # meta-methods
    def __init__(self, token, **kwds):
        # chain up
        super().__init__(**kwds)
        # extract my name
        self.name = [fragment.strip() for fragment in token.lexeme.split(self.scopeSeparator)]
        # save the source of the token
        self.locator = token.locator
        # all done
        return


# end of file
