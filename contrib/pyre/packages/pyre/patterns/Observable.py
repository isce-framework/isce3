# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


import weakref


class Observable:
    """
    Provide notification support for classes that maintain dynamic associations with multiple
    clients.

    Observers, i.e. clients of the Observable, register event handlers that will be invoked to
    notify them whenever something interesting happens to the Observable. The nature of what is
    being observed is defined by Observable descendants and their managers. For example,
    instances of pyre.calc.Node are observable by the other nodes whose value depends on
    them. These dependents are notified of value changes and forced to refresh their value
    cache.

    The event handlers are callables that take the observable instance as their single
    argument.

    interface:
      addObserver: registers its callable argument with the list of handlers to invoke
      removeObserver: remove an event handler from the list of handlers to invoke
      notifyObservers: invoke the registered handlers in the order in which they were registered

    """


    # public data
    observers = None


    # interface
    def notifyObservers(self, **kwds):
        """
        Notify all observers
        """
        # print("pyre.patterns.Observable: {}".format(self))
        # build a list before notification, just in case the observer's callback behavior
        # involves removing itself from our callback set
        for instance, method in tuple(self.observers.items()):
            # print("  notifying {}".format(instance))
            # invoke the callable
            method(instance, observable=self, **kwds)
        # all done
        return self


    # callback management
    def addObserver(self, callback):
        """
        Add {callback} to the set of observers
        """
        # extract the caller information from the method
        instance = callback.__self__
        method = callback.__func__
        # update the observers
        self.observers[instance] = method
        # and return
        return self


    def removeObserver(self, callback):
        """
        Remove {callback} from the set of observers
        """
        # extract the caller information from the method
        instance = callback.__self__
        # remove this observer
        del self.observers[instance]
        # and return
        return self


    # meta methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # initialize the pile of observers
        self.observers = weakref.WeakKeyDictionary()
        # all done
        return


# end of file
