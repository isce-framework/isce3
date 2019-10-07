# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import weakref
# the superclas
from .Reactor import  Reactor


# class declaration
class Observable(Reactor):
    """
    Mix-in class that notifies its clients when the value of a node changes
    """

    # public data
    @property
    def observers(self):
        """
        Return an iterable over my live observers
        """
        # go through the references to my observers
        for ref in self._observers:
            # unwrap it
            observer = ref()
            # if it is dead
            if observer is None:
                # skip it
                continue
            # otherwise, send it along
            yield observer

        # all done
        return


    # observer management
    def addObserver(self, observer):
        """
        Add {observer} to my pile
        """
        # build a weak reference to {observer} and add it to the pile
        self._observers.add(weakref.ref(observer))
        # all done
        return self


    def removeObserver(self, observer):
        """
        Remove {observer} from my pile
        """
        # build a weak reference to {observer} and remove it from the pile
        self._observers.remove(weakref.ref(observer))
        # all done
        return self


    def replace(self, obsolete):
        """
        Remove {obsolete} from its upstream graph and assume its responsibilities
        """
        # make a pile of nodes that have been adjusted
        clean = set()
        # go through the observers of {obsolete}
        for observer in tuple(obsolete.observers):
            # remove it from the pile in {obsolete}
            obsolete.removeObserver(observer)
            # if the observer is a composite
            if isinstance(observer, self.composite):
                # perform a substitution
                observer.substitute(current=obsolete, replacement=self, clean=clean)
            # add it to my pile
            self.addObserver(observer)
            # and let it know things have changed
            observer.flush(observable=self)
        # all done
        return super().replace(obsolete=obsolete)


    # signaling
    def flush(self, **kwds):
        """
        Handler of the notification event from one of my observables
       """
        # take this opportunity to clean up my observers
        observers = set(self.observers)
        # go through the pile
        for observer in observers:
            # notify each one
            observer.flush(observable=self)
        # rebuild the set of references
        self._observers = set(map(weakref.ref, observers))
        # chain up
        return super().flush(**kwds)


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # initialize the set of my observers
        self._observers = set() # should have been a weak set, but I can do better...
        # all done
        return


# end of file
