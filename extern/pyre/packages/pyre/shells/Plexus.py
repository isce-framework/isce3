# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the framework
import pyre
# superclass
from .Application import Application


# class declaration
class Plexus(Application):
    """
    Base class for applications that interpret their first non-configurational command line
    argument as an action that should be executed.

    As an example of such an app, consider {merlin}. Invoking

        merlin version

    causes the {merlin} plexus to locate an action named {version} and invoke it.

    Subclasses are expected to declare a class variable {pyre_action} that points to a subclass
    of the {Action} protocol defined in this package. This responsibility is left to subclasses
    so that there is no bias on the category names of the actions induced by a choice of family
    name by the framework. This way the language used to specify the actions and their behavior
    can remain natural to the context of the application.
    """


    # interface
    @pyre.export
    def main(self, *args, **kwds):
        """
        The plexus main entry point interprets the first non-configurational command line argument
        as the name of an action to perform
        """
        # grab my command line arguments
        argv = self.argv
        # attempt to
        try:
            # get the name of the command
            name = next(argv)
        # if there aren't any
        except StopIteration:
            # we will deal with this case later; it is important to do as little as possible
            # here so we can exit the exception block quickly and cleanly
            pass
        # if there is something to invoke
        else:
            # do it
            return self.pyre_invoke(action=name, argv=argv)

        # otherwise, just show the help screen
        return self.help()


    @pyre.export
    def help(self, **kwds):
        """
        Hook for the application help system
        """
        # grab my command line arguments
        argv = self.argv
        # and attempt to
        try:
            # get the name of a command
            name = next(argv)
        # if there wasn't one
        except StopIteration:
            # no worries, we handle this case below
            pass
        # if there was one
        else:
            # we have the name of a command; resolve it
            command = self.pyre_repertoir.resolve(plexus=self, spec=name)
            # and invoke its help
            return command.help(plexus=self)

        # if we get this far, the user has asked for help without specifying a particular
        # command; we display the general help screen

        # show the application banner
        self.info.line(self.pyre_banner())
        # show help on available actions
        self.pyre_documentActions()
        # flush
        self.info.log()
        # and indicate success
        return 0


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # build my repertoir
        self.pyre_repertoir = self.pyre_newRepertoir()
        # all done
        return


    # implementation details
    # hooks
    def pyre_newRepertoir(self):
        """
        Factory for the manager of actions
        """
        # get the default implementation
        from .Repertoir import Repertoir
        # build one and return it
        return Repertoir(protocol=self.pyre_action)


    # convenience
    def pyre_invoke(self, action, argv):
        """
        Invoke the named {action} with {argv} as additional arguments
        """
        # resolve and invoke; typos and such get handled by {pyre_repertoir}
        return self.pyre_repertoir.invoke(plexus=self, action=action, argv=argv)


    # support for the help system
    def pyre_documentActions(self, indent=' '*4):
        """
        Place information about the available actions in my {info} channel
        """
        # reset the pile of actions
        actions = []
        # get the documented commands
        for uri, name, action, tip in self.pyre_action.pyre_documentedActions():
            # and put them on the pile
            actions.append((name, tip))
        # if there were any
        if actions:
            # figure out how much space we need
            width = max(len(name) for name, _ in actions)
            # introduce this section
            self.info.line('commands:')
            # for each documented action
            for name, tip in actions:
                # show the details
                self.info.line('{}{:>{}}: {}'.format(indent, name, width, tip))
            # some space
            self.info.line()

        # all done
        return


    # data
    pyre_repertoir = None


# end of file
