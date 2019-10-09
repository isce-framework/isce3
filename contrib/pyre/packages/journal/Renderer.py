# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# packages
import pyre


# declaration
class Renderer(pyre.protocol, family="journal.renderers"):
    """
    The protocol specification that renderers must satisfy
    """


    # my default implementation
    @classmethod
    def pyre_default(cls, **kwds):
        """
        Examine {sys.stdout} and turn on color output if the current terminal type supports it
        """
        # access the stdout stream
        import sys
        # if it is a tty
        try:
            if sys.stdout.isatty():
                # figure out the terminal type
                import os
                term = os.environ.get('TERM', 'unknown').lower()
                # if it is ANSI compatible
                if term in cls.ansi:
                    # the default is colored
                    from .ANSIRenderer import ANSIRenderer
                    return ANSIRenderer
        # some devices don't support isatty
        except AttributeError:
            pass
        # plain text, by default
        from .TextRenderer import TextRenderer
        return TextRenderer


    # interface
    @pyre.provides
    def render(self, text, metadata):
        """
        Convert the diagnostic information into a form that a device can record
        """


    # private data
    ansi = {'ansi', 'vt102', 'vt220', 'vt320', 'vt420', 'xterm', 'xterm-color'}


# end of file
