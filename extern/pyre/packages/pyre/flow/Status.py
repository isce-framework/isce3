# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# support
import pyre


# declaration
class Status(pyre.tracker):
    """
    A helper that watches over a component's traits and records value changes
    """


    # public data
    @property
    def stale(self):
        """
        Return my current status
        """
        # easy enough
        return self._stale

    @stale.setter
    def stale(self, status):
        """
        Adjust my status
        """
        # if i'm being marked as stale
        if status is True:
            # flush
            return self.flush()
        # otherwise, just update the status
        self._stale = status
        # all done
        return self


    # interface
    def playback(self, node, alias):
        """
        Go through the history of the trait named {alias}
        """
        # find its trait by this name
        trait = node.pyre_trait(alias=alias)
        # get the key
        key = node.pyre_inventory[trait].key
        # chain up
        yield from super().playback(key=key)
        # all done
        return


    # input binding
    def addInputBinding(self, **kwds):
        """
        The given {product} is now an input to {factory}
        """
        # show me
        # self.log(activity="adding input", **kwds)
        # all done
        return self


    def removeInputBinding(self, **kwds):
        """
        The given {product} is no longer an input to {factory}
        """
        # show me
        # self.log(activity="removing input", **kwds)
        # all done
        return self


    # output binding
    def addOutputBinding(self, **kwds):
        """
        The given {product} is now an output of {factory}
        """
        # show me
        # self.log(activity="adding output", **kwds)
        # all done
        return self


    def removeOutputBinding(self, **kwds):
        """
        The given {product} is no longer an output of {factory}
        """
        # show me
        self.log(activity="removing output", **kwds)
        # all done
        return self


    # meta-methods
    def __init__(self, node, stale=False, **kwds):
        # chain up
        super().__init__(**kwds)
        # initialize my flag
        self._stale = stale
        # enable tracking
        self.track(component=node)
        # all done
        return


    # hooks
    def flush(self, **kwds):
        """
        Handler of the notification that the value of an {observable} has changed
        """
        # update my state
        self._stale = True
        # chain up
        return super().flush(**kwds)


    # implementation details
    def log(self, activity, factory, product):
        # show me
        print(f"pyre.flow.Status: {activity}")
        print(f"  status: {self}")
        print(f"  factory: {factory}")
        print(f"  product: {product}")
        print(f"")
        # all done
        return


    # private data
    _stale = None


# end of file
