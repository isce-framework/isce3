#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify direct access to the namespace
"""


def test():
    # access the package
    import pyre
    # and the nameserver
    nameserver = pyre.executive.nameserver

    # first batch of names
    alias = 'first'
    canonical = nameserver.join('package.application', alias)
    # alias them
    nameserver.alias(target=canonical, alias=alias)
    # hash them
    aliasKey = nameserver.hash(alias)
    canonicalKey = nameserver.hash(canonical)
    # verify that both names now hash to the same key
    assert aliasKey == canonicalKey
    # at this point, the nodes map should not know the keys
    assert nameserver._nodes.get(aliasKey) == None
    assert nameserver._nodes.get(canonicalKey) == None
    # and neither should the name map
    assert nameserver._metadata.get(aliasKey) == None
    assert nameserver._metadata.get(canonicalKey) == None

    # next batch
    alias = 'second'
    canonical = nameserver.join('package.application', alias)
    # add a value under the canonical name
    nameserver[canonical] = canonical
    # alias them
    nameserver.alias(target=canonical, alias=alias)
    # hash them
    aliasKey = nameserver.hash(alias)
    canonicalKey = nameserver.hash(canonical)
    # verify that both names now hash to the same key
    assert aliasKey == canonicalKey
    # the node map should have our value under the canonical key
    assert nameserver._nodes[canonicalKey].value == canonical
    # the canonical key should map to its name
    assert nameserver._metadata[canonicalKey].name == canonical

    # invert the relationship: existing alias, new canonical
    alias = 'third'
    canonical = nameserver.join('package.application', alias)
    # add a value under the alias
    nameserver[alias] = alias
    # alias them
    nameserver.alias(target=canonical, alias=alias)
    # hash them
    aliasKey = nameserver.hash(alias)
    canonicalKey = nameserver.hash(canonical)
    # verify that both names now hash to the same key
    assert aliasKey == canonicalKey
    # the node map should have our value under the canonical key
    assert nameserver._nodes[canonicalKey].value == alias
    # the canonical key should map to its name
    assert nameserver._metadata[canonicalKey].name == canonical

    # both pre-existing, with canonical having higher priority
    alias = 'fourth'
    canonical = nameserver.join('package.application', alias)
    # add a value under the alias
    nameserver[alias] = alias
    # and a value under the canonical name
    nameserver[canonical] = canonical
    # alias them
    nameserver.alias(target=canonical, alias=alias)
    # hash them
    aliasKey = nameserver.hash(alias)
    canonicalKey = nameserver.hash(canonical)
    # verify that both names now hash to the same key
    assert aliasKey == canonicalKey
    # the node map should have our value under the canonical key
    assert nameserver._nodes[canonicalKey].value == canonical
    # the canonical key should map to its name
    assert nameserver._metadata[canonicalKey].name == canonical

    # both pre-existing, with the alias having higher priority
    alias = 'fifth'
    canonical = nameserver.join('package.application', alias)
    # add a value under the canonical name
    nameserver[canonical] = canonical
    # and add a value under the alias
    nameserver[alias] = alias
    # alias them
    nameserver.alias(target=canonical, alias=alias)
    # hash them
    aliasKey = nameserver.hash(alias)
    canonicalKey = nameserver.hash(canonical)
    # verify that both names now hash to the same key
    assert aliasKey == canonicalKey
    # the node map should have our value under the canonical key
    assert nameserver._nodes[canonicalKey].value == alias
    # the canonical key should map to its name
    assert nameserver._metadata[canonicalKey].name == canonical

    # and return the managers
    return nameserver


# main
if __name__ == "__main__":
    test()


# end of file
