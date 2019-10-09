# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the framework
import pyre
# my base class
from .Script import Script


class Interactive(Script, family="pyre.shells.interactive"):
    """
    A shell that invokes the main application behavior and then enters interactive mode
    """


    # interface
    @pyre.export
    def launch(self, application, *args, **kwds):
        """
        Invoke the application behavior
        """
        # show the application help screen
        application.help()
        # enter interactive mode
        status = self.pyre_interactiveSession(application=application)
        # all done
        return status


    # implementation details
    def pyre_interactiveSession(self, application=None, banner=None, context=None):
        """
        Convert this session to an interactive one
        """
        # we need an application specific tag
        tag = (application.pyre_namespace or application.pyre_name) if application else 'pyre'

        # go live
        import code, sys
        # configure {readline}
        self.pyre_enrich(tag)
        # adjust the prompts
        sys.ps1 = f'{tag}: '
        sys.ps2 = '  ... '

        # prime the local namespace
        symbols = {}
        # if we have an application instance
        if application:
            # provide access to the application object itself
            symbols['app'] = application
            # and ask it to decorate further
            symbols = application.pyre_interactiveSessionContext(context=symbols)
        # if the caller gave us {context}
        if context:
            # include it
            symbols.update(context)

        # if we are serving a specific application
        if banner is None and application:
            # ask it for a sign on banner
            banner = application.pyre_interactiveBanner()

        # enter interactive mode
        return code.interact(banner=banner, local=symbols, exitmsg=f'{tag}: exiting...')


    def pyre_enrich(self, tag):
        """
        Attempt to provide a richer interactive experience
        """
        # attempt to
        try:
            # pull readline support
            import readline, rlcompleter
        # if unable
        except ImportError:
            # is there anything else we can do?
            return

        # get the package docstring, gingerly
        doc = getattr(readline, '__doc__', None)
        # on OSX, {readline} may be implemented using {libedit}
        if 'libedit' in doc:
            # bind the <tab> character to the correct behavior
            readline.parse_and_bind('bind ^I rl_complete')
        # otherwise
        else:
            # assume gnu {readline}
            readline.parse_and_bind('tab: complete')

        # again carefully, try to
        try:
            # initialize support
            readline.read_init_file()
        # if that fails
        except OSError:
            # there are many possible causes; most are unlikely on a well managed system
            # in all likelihood, it's just that the configuration file doesn't exist
            # can't do much about any of them, anyway
            pass

        # build the uri to the history file
        history = pyre.primitives.path('~', f'.{tag}-history').expanduser().resolve()
        # stringify
        history = str(history)
        # attempt to
        try:
            # read it
            readline.read_history_file(history)
        # if not there
        except IOError:
            # no problem
            pass
        # make sure it gets saved
        import atexit
        # by registering a handler for when the session terminates
        atexit.register(readline.write_history_file, history)

        # all done
        return


# end of file
