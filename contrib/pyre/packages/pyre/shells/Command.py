# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import itertools
# access to the framework
import pyre
# my protocol
from .Action import Action


# class declaration
class Command(pyre.component, implements=Action):
    """
    A component that implements {Action}
    """


    # public state
    dry = pyre.properties.bool(default=False)
    dry.doc = "show what would get done without actually doing anything"


    # expected interface
    @pyre.export
    def main(self, plexus, **kwds):
        """
        This is the implementation of the action
        """
        # just print a message
        plexus.info.log('main: missing implementation')
        # and indicate success
        return 0


    @pyre.export(tip='show this help screen')
    def help(self, plexus, **kwds):
        """
        Show a help screen
        """
        # indentation level
        indent = '    '
        # my specification
        spec = '{.pyre_namespace} {.pyre_spec}'.format(plexus, self)
        # tell the user what they typed
        plexus.info.line(spec)
        # generate a simple help screen
        for line in self.pyre_help(spec=spec, indent=indent):
            # and push it to my info channel
            plexus.info.line(line)
        # flush
        plexus.info.log()
        # and indicate success
        return 0


    # meta-methods
    def __init__(self, name, spec, plexus, **kwds):
        # chain up
        super().__init__(name=name, **kwds)
        # save my short name
        self.pyre_spec = spec
        # all done
        return


    # implementation details
    def __call__(self, plexus, argv):
        """
        Commands are callable
        """
        # delegate to {main}
        return self.main(plexus=plexus, argv=argv)


    def pyre_help(self, spec, indent=' '*4, **kwds):
        """
        Hook for the application help system
        """
        # my summary
        yield from self.pyre_showSummary(indent=indent, **kwds)
        # my behaviors
        yield from self.pyre_showBehaviors(spec=spec, indent=indent, **kwds)
        # my public state
        yield from self.pyre_showConfigurables(indent=indent, **kwds)
        # all done
        return


    # private data
    pyre_spec = None


# end of file
