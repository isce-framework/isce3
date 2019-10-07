# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# class declaration
class Reactor:
    """
    Abstract base class that supplies the default implementation of a single method

    The purpose of this class is decouple classes that would otherwise have to derive from each
    other. As an example, consider {Observer} and {Observable} as mixins that provide support
    for their corresponding aspects of the {Observer} pattern. Things are fairly
    straightforward until a class is an observable observer, which forces a specific hierarchy
    in order to implement {flush} correctly. {Reactor} helps solve this problem by providing an
    empty {flush} method that subclasses can chain up to without having to worry about their
    {mro}.
    """


    # signaling
    def flush(self, **kwds):
        """
        Handler of notification events
        """
        # nothing to do
        return self


# end of file
