# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the framework
import pyre
# my base class
from .Executive import Executive


class Script(Executive, family="pyre.shells.script"):
    """
    A shell that invokes the main application behavior and then exits
    """

    # user configurable state
    # the help markers
    helpon = pyre.properties.strings(default=['?', 'h', 'help'])
    helpon.doc = "the list of markers that indicate the user has asked for help"

    # a marker that enables applications to deduce the type of shell that is hosting them
    model = pyre.properties.str(default='script')
    model.doc = "the programming model"


    # interface
    @pyre.export
    def launch(self, application, *args, **kwds):
        """
        Invoke the application behavior
        """
        # the only decision to make here is whether to invoke the help system;
        # get the nameserver
        nameserver = self.pyre_nameserver
        # go through the markers
        for marker in self.helpon:
            # if it is known by the configuration store
            if marker in nameserver:
                # get help
                return application.help(*args, **kwds)

        # set up a net
        try:
            # launch the application
            status = application.main(*args, **kwds)
        # if the user interrupted
        except KeyboardInterrupt as event:
            # launch the handler
            status = application.pyre_interrupted(info=event)
        # if the framework complained about something
        except self.PyreError as error:
            # if we are in debug mode
            if application.DEBUG:
                # let the error through
                raise
            # otherwise, log it
            application.error.log(str(error))
            # indicate a failure
            status = 1
        # if all ended well
        else:
            # shutdown
            application.pyre_shutdown(status=status)

        # in any case, we are all done
        return status


# end of file
