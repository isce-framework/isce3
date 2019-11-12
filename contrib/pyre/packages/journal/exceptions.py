# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# get the base pyre exception
from pyre.framework.exceptions import PyreError


# firewalls
class FirewallError(PyreError):
    """
    Exception raised whenever a fatal firewall is encountered
    """

    # public data
    description = "firewall breached; aborting..."

    # meta-methods
    def __init__(self, firewall, **kwds):
        # chain up
        super().__init__(locator=firewall.locator, **kwds)
        # record the error
        self.firewall = firewall
        # all done
        return


# application errors
class ApplicationError(PyreError):
    """
    Exception raised whenever an application error is encountered
    """


    # public data
    description = "firewall breached; aborting..."

    # meta-methods
    def __init__(self, error, **kwds):
        # chain up
        super().__init__(locator=error.locator, **kwds)
        # record the error
        self.error = error
        # all done
        return


# end of file
