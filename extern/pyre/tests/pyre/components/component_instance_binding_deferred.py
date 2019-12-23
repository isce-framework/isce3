#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


import pyre

class ifac(pyre.protocol, family="deferred.ifac"):
    """sample protocol"""
    @classmethod
    def pyre_default(cls, **kwds): return comp

class comp(pyre.component, family="deferred.ifac.comp", implements=ifac):
    """an implementation"""
    tag = pyre.properties.str()

class user(pyre.component, family="deferred.user"):
    """a component user"""
    comp = ifac()

class container(pyre.component, family="deferred.container"):
    """a component container"""
    name = pyre.properties.str(default=None)
    comp = ifac()
    catalog = pyre.properties.dict(schema=ifac())


def test():
    # build the individual user
    u = user(name="user")
    # verify that its {comp} is configured correctly
    assert u.comp.tag == "one"

    # build the shell
    s = container(name="tagger")
    # verify that the catalog has three members
    assert len(s.catalog) == 3
    # and that the contents were configured properly
    for name, instance in s.catalog.items():
        # by matching the component tag against its name
        assert instance.tag == name

    # all done
    return s


# main
if __name__ == "__main__":
    test()


# end of file
