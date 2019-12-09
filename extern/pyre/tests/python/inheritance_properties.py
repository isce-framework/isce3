#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that properties and other descriptors work as expected
"""


def test():
    # the setup
    class descriptor(object):

        def __get__(self, instance, cls=None):
            if instance: return getattr(instance, "marker")
            return getattr(cls, "marker")

    class base(object):

        marker = "base"
        dscr = descriptor()

    class derived(base):

        marker = "derived"
        def __init__(self): self.marker = "instance of derived"


    # verify
    assert base.dscr == "base"
    assert derived.dscr == "derived"

    instance = derived()
    assert instance.dscr == "instance of derived"

    return


# main
if __name__ == "__main__":
    test()


# end of file
