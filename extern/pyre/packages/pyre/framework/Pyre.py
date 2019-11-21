# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import weakref
from .. import tracking
# superclass
from .Executive import Executive


# declaration
class Pyre(Executive):
    """
    The default framework executive.

    This class is responsible for the actual instantiation of the providers of the framework
    services.
    """


    # constants
    locator = tracking.simple('during pyre startup')

    # public data
    from . import _verbose as verbose


    # interface
    def activate(self, **kwds):
        """
        Initialize the providers of the runtime services
        """
        # chain up to my base class
        super().activate(**kwds)

        # access the command line
        import sys
        # make a parser
        parser = self.newCommandLineParser()
        # parse the command line
        events = parser.parse(argv=sys.argv[1:])
        # ask my configurator to process the configuration events
        self.configurator.processEvents(events=events, priority=self.priority.command)

        # all done
        return self


    def check(self):
        """
        Report and boot time errors
        """
        # bail out if no errors were detected
        if not self.errors: return

        # report the boot time errors
        # MGA: ??can i use journal??
        print(' ** pyre: the following errors were encountered while booting:')
        for error in self.errors:
            print(' ++   {}'.format(error))

        # all done
        return self


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)

        # get storage for the framework manager proxies
        from .Dashboard import Dashboard as dashboard

        # attach me
        dashboard.pyre_executive = weakref.proxy(self)

        # build my nameserver
        self.nameserver = self.newNameServer()
        # attach
        dashboard.pyre_nameserver = weakref.proxy(self.nameserver)

        # my fileserver
        self.fileserver = self.newFileServer(executive=self)
        # attach
        dashboard.pyre_fileserver = weakref.proxy(self.fileserver)

        # component bookkeeping
        self.registrar = self.newComponentRegistrar()
        # attach
        dashboard.pyre_registrar = weakref.proxy(self.registrar)

        # handler of configuration events
        self.configurator = self.newConfigurator(executive=self)
        # attach
        dashboard.pyre_configurator = weakref.proxy(self.configurator)

        # database schema
        self.schema = self.newSchema(executive=self)
        # attach
        dashboard.pyre_schema = weakref.proxy(self.schema)

        # component linker
        self.linker = self.newLinker()
        # the timer registry
        self.timekeeper = self.newTimerRegistry()

        # all done
        return


# end of file
