# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# declaration
class Dashboard:
    """
    Mix-in class that provides access to the pyre executive and its managers
    """

    # grab the base of all pyre exceptions
    from .exceptions import PyreError


    # public data
    # the executive
    pyre_executive = None

    # framework parts
    pyre_fileserver = None
    pyre_nameserver = None
    pyre_configurator = None

    # infrastructure managers
    pyre_registrar = None # the component registrar
    pyre_schema = None # the database schema

    # information about the runtime environment
    pyre_host = None # the current host
    pyre_user = None # the current user
    pyre_application = None # the current application


    # debugging support
    @classmethod
    def dashboard(cls):
        """
        Dump the status of the dashboard
        """
        # show me
        yield "executive: {.pyre_executive}".format(cls)
        yield "  fileserver: {.pyre_fileserver}".format(cls)
        yield "  nameserver: {.pyre_nameserver}".format(cls)
        yield "  configurator: {.pyre_configurator}".format(cls)
        yield "  registrar: {.pyre_registrar}".format(cls)
        yield "  schema: {.pyre_schema}".format(cls)
        yield "  host: {.pyre_host}".format(cls)
        yield "  user: {.pyre_user}".format(cls)
        yield "  application: {.pyre_application}".format(cls)
        # all done
        return


# end of file
