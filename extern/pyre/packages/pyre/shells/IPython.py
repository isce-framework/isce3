# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the framework
import pyre
# my base class
from .Interactive import Interactive


class IPython(Interactive, family="pyre.shells.ipython"):
    """
    A shell that invokes the main application behavior and then enters IPython mode
    """


    # implementation details
    def pyre_interactiveSession(self, application, context=None):
        """
        Convert this session to an interactive one
        """
        # attempt to
        try:
            # get support for IPython
            import IPython
        # if this fails
        except ImportError:
            # print an error message
            application.error.log("could not import the IPython module; is it installed?")
            # and bail
            raise SystemExit(1)

        # otherwise, prime the local namespace
        context = context or {}
        # adjust it
        context['app'] = application
        # give the application an opportunity to add symbols as well
        context = application.pyre_interactiveSessionContext(context=context)
        # enter interactive mode
        return IPython.start_ipython(argv=[], user_ns=context)


# end of file
