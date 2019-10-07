# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#



# class declaration
class Memo:
    """
    A mix-in class that implements value memoization
    """


    # public data
    dirty = True


    # interface
    def getValue(self, **kwds):
        """
        Override the node value retriever and return the contents of my value cache if it is up
        to date; otherwise, recompute the value and update the cache
        """
        # if my cache is invalid
        if self.dirty:
            # recompute
            self._value = super().getValue(**kwds)
            # mark
            self.dirty = False
        # return the cache contents
        return self._value


    def setValue(self, value, **kwds):
        """
        Override the value setter to refresh my cache and notify my observers
        """
        # update the value
        super().setValue(value=value, **kwds)
        # mark me as dirty
        self.dirty = True
        # all done
        return self


    # cache management
    def flush(self, **kwds):
        """
        Invalidate my cache and notify my observers
        """
        # do nothing if my cache is already invalid
        if self.dirty: return self
        # otherwise, invalidate the cache
        self.dirty = True
        # and notify my observers
        return super().flush(**kwds)


    # private data
    _value = None


# end of file
