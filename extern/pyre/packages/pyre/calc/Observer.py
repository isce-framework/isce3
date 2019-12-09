# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# the superclass
from .Reactor import Reactor
# the complement
from .Observable import Observable


# class declaration
class Observer(Reactor):
    """
    Mix-in class that enables a node to be notified when the value of its dependents change
    """


    # interface
    def observe(self, observables):
        """
        Add me as an observer to all {observables}
        """
        # loop through {observables}
        for observable in observables:
            # skip the ones that are not observable
            if not isinstance(observable, Observable): continue
            # add me as an observer to the rest
            observable.addObserver(self)
        # all done
        return self


    def ignore(self, observables):
        """
        Stop observing the {observables}
        """
        # loop through {observables}
        for observable in observables:
            # skip the ones that are not observable
            if not isinstance(observable, Observable): continue
            # drop me as an observer from the rest
            observable.removeObserver(self)
        # all done
        return self


# end of file
