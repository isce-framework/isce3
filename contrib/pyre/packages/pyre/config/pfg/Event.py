# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# the base class for all configuration event handlers
class Event:
    """
    The abstract base class for all configuration event handlers
    """


    # constants
    scopeSeparator = '.'
    fragmentSeparator = '#'

    # pull the configuration event types
    from .. import events


    # interface
    def notify(self, parent):
        """
        Invoked when configuration data gathering is done for the active node and it is time to
        delegate any further processing to the containing node
        """
        # abstract
        raise NotImplementedError(
            "class {.__name__!r} must implement 'notify'".format(type(self)))


# end of file
