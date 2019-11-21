#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the implementation of Observable works as advertised
"""


def test():
    from pyre.patterns.Observable import Observable

    class node(Observable):
        """
        the Observable
        """

        @property
        def value(self):
            return self._value

        @value.setter
        def value(self, value):
            self._value = value
            self.notifyObservers()
            return self

        def __init__(self, value, **kwds):
            super().__init__(**kwds)
            self._value = value
            return

    class probe:
        """
        the observer
        """

        def __init__(self, node, **kwds):
            super().__init__(**kwds)
            self.cache = None
            node.addObserver(self._update)
            return

        def _update(self, observable, **kwds):
            self.cache = observable.value
            return

    n = node(0)
    p = probe(n)
    n.value = 3.14159

    assert p.cache == n.value

    return node, probe


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
