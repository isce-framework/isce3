#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that probes get notified when the values of their nodes change
"""


# tuck all the object references in a function so they get a chance to go out of scope
def test():
    import pyre.calc

    # a probe is an observer
    from pyre.calc.Probe import Probe
    # make a probe that records the values of the monitored nodes
    class probe(Probe):

        def flush(self, observable):
            self.nodes[observable] = observable.value
            return self

        def __init__(self, **kwds):
            super().__init__(**kwds)
            self.nodes = {}
            return

    # make some nodes
    u = pyre.calc.var(value='u')
    v = pyre.calc.var(value='v')
    w = pyre.calc.var(value='w')
    s = u + v
    r = u.ref()

    # as we are, {u} has two observers: {s} and {r}
    assert identical(u.observers, {r,s})

    # make a probe
    p = probe()
    # ask it watch {u} and its dependents
    p.observe(observables=(u,s,r))

    # {u} now has three observers: {s}, {r}, and {p}
    # print(set(u.observers))
    assert identical(u.observers, {r,s,p})

    # {w} should have no observers
    assert identical(w.observers, {})
    # make the substitution
    w.replace(u)
    # print(set(w.observers))
    # print(set(u.observers))
    # {w} should have three observers: {s}, {r}, and {p}
    assert identical(w.observers, {p,s,r})

    return


def identical(s1, s2):
    """
    Verify that the nodes in {s1} and {s2} are identical. This has to be done carefully since
    we must avoid triggering __eq__
    """
    # realize both of them
    s1 = tuple(s1)
    s2 = tuple(s2)
    # fail if their lengths are not the same
    if len(s1) != len(s2): return False
    # go through one
    for n1 in s1:
        # and the other
        for n2 in s2:
            # if this is a match, we are done
            if n1 is n2: break
        # if we didn't n1 in s2
        else:
            # fail
            return False
    # if we get this far, all is good
    return True


# main
if __name__ == "__main__":
    # request debugging support for the pyre.calc package
    pyre_debug = { "pyre.calc" }
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # run the test
    test()
    # verify reference counts
    from pyre.calc.Node import Node
    # print(tuple(Node._pyre_extent))
    assert tuple(Node._pyre_extent) == ()


# end of file
