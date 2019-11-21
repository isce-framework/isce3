#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


import pyre

class ifac(pyre.protocol, family="sample.ifac"):
    """sample protocol"""

class comp(pyre.component, family="sample.ifac.comp", implements=ifac):
    """an implementation"""
    tag = pyre.properties.str()

class container(pyre.component, family="sample.container"):
    """a component container"""
    catalog = pyre.properties.dict(schema=ifac())


def test():
    # build the shell
    s = container(name="catalog_container")
    # verify that the catalog has three members
    # print(len(s.catalog))
    assert len(s.catalog) == 3
    # and that the contents were configured properly
    for name, instance in s.catalog.items():
        # print("tag: {!r}, name: {!r}".format(instance.tag, name))
        assert instance.tag == name

    return s


# main
if __name__ == "__main__":
    test()


# end of file
