# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Task:
    """
    Base class for functors that are part of an application data model
    """


    # types
    # easy access to the base indicator of a temporary failure during task execution
    from .exceptions import RecoverableError
    # access to status enums for this task and its broader execution context
    from .CrewStatus import CrewStatus as crewcodes
    from .TaskStatus import TaskStatus as taskcodes


    # interface
    def execute(self, **kwds):
        """
        The body of the functor
        """
        # N.B.: defined to support subclasses that employ cooperative multiple inheritance;
        # they can chain up to this empty implementation
        # nothing to do
        return


    # meta-methods
    def __call__(self, **kwds):
        # forward to the {execute} method
        return self.execute(**kwds)


# end of file
