# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# look for the timer extension and adjust the interface accordingly
try:
    # import the extension module
    from .NativeTimer import NativeTimer as newTimer
# if the module is not available, fall back to the pure python implementation
except ImportError:
    from .PythonTimer import PythonTimer as newTimer


# declaration
class Registrar:
    """
    This class maintains the association between names and timers.

    Registrar acts a timer factory: upon the first use of a given timer name, it builds a timer
    and registers it with its internal index. Subsequent requests for timers by the same name
    return the original instance, effectively making timers accessible in non-local ways. See
    the documentation for {pyre.timers} for more information and simple examples.
    """


    # interface
    def timer(self, name, **kwds):
        """
        Build and register a new timer under {name}, if this is the first time {name} is
        used. Otherwise, retrieve the named timer
        """
        # attempt to retrieve the named timer
        try:
            timer = self._index[name]
        # if it doesn't exist
        except KeyError:
            # build a new one
            timer = newTimer(name=name, **kwds)
            # and register it
            self._index[name] = timer
        # all done
        return timer


    # meta methods
    def __init__(self, **kwds):
        # chain to the ancestors
        super().__init__(**kwds)

        # initialize the timer index
        self._index = dict()

        # all done
        return


    # implementation details
    _index = None


# end of file
