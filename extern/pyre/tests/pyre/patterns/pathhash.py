#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Check that path hashes work as advertised
"""


def test():
    # access the class
    import pyre.patterns
    # build one
    pathhash = pyre.patterns.newPathHash()
    # here are a couple of multi-level addresses
    separator = '.'
    moduleName = "pyre.patterns.PathHash".split(separator)
    klassName = moduleName + ["PathHash"]

    # now hash the matching nodes
    module = pathhash.hash(moduleName)
    klass = pathhash.hash(items=klassName)
    # check that i get the same node the second time i retrieve it
    assert module == pathhash.hash(items=moduleName)
    assert klass == pathhash.hash(items=klassName)
    # check that i can retrieve the class from within the module
    assert klass == module.hash(items=["PathHash"])

    # build an alias for the module
    base = pathhash.hash(items=['pyre'])
    alias = "pathhash"
    original = pathhash.hash(items="pyre.patterns.PathHash".split(separator))

    base.alias(alias=alias, target=original)
    # check that the alias points where it should
    assert module == pathhash.hash(items="pyre.pathhash".split(separator))
    # and that both contain the same class
    assert klass == pathhash.hash(items="pyre.pathhash.PathHash".split(separator))

    # dump out the contents of the hash
    # pathhash.dump()

    # return the pathhash
    return pathhash


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
