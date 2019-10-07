#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise the URI parser
"""


def test():
    import pyre.descriptors

    # make a converter
    uri = pyre.descriptors.uri()

    # the canonical case
    parts = uri.coerce("scheme://authority/address?query#fragment")
    assert parts.scheme == 'scheme'
    assert parts.authority == 'authority'
    assert parts.address == '/address'
    assert parts.query == 'query'
    assert parts.fragment == 'fragment'

    # drop the fragment
    parts = uri.coerce("scheme://authority/address?query")
    assert parts.scheme == 'scheme'
    assert parts.authority == 'authority'
    assert parts.address == '/address'
    assert parts.query == 'query'
    assert parts.fragment == None

    # drop the query
    parts = uri.coerce("scheme://authority/address#fragment")
    assert parts.scheme == 'scheme'
    assert parts.authority == 'authority'
    assert parts.address == '/address'
    assert parts.query == None
    assert parts.fragment == 'fragment'

    # drop both the query and the fragment
    parts = uri.coerce("scheme://authority/address")
    assert parts.scheme == 'scheme'
    assert parts.authority == 'authority'
    assert parts.address == '/address'
    assert parts.query == None
    assert parts.fragment == None

    # drop the fragment, the query and the authority, with an absolute address
    parts = uri.coerce("scheme:/address")
    assert parts.scheme == 'scheme'
    assert parts.authority == None
    assert parts.address == '/address'
    assert parts.query == None
    assert parts.fragment == None

    # drop the fragment, the query and the authority, with a relative address
    parts = uri.coerce("scheme:address")
    assert parts.scheme == 'scheme'
    assert parts.authority == None
    assert parts.address == 'address'
    assert parts.query == None
    assert parts.fragment == None

    # drop the fragment, the query and the authority, with a multi-level absolute address
    parts = uri.coerce("scheme:/addr1/addr2")
    assert parts.scheme == 'scheme'
    assert parts.authority == None
    assert parts.address == '/addr1/addr2'
    assert parts.query == None
    assert parts.fragment == None

    # drop the fragment, the query and the authority, with a multi-level relative address
    parts = uri.coerce("scheme:addr1/addr2")
    assert parts.scheme == 'scheme'
    assert parts.authority == None
    assert parts.address == 'addr1/addr2'
    assert parts.query == None
    assert parts.fragment == None

    # a simple case
    parts = uri.coerce("pyre.pml")
    assert parts.scheme == None
    assert parts.address == 'pyre.pml'
    assert parts.query == None
    assert parts.fragment == None

    # another simple case
    parts = uri.coerce("/pyre/system/pyre.pml")
    assert parts.scheme == None
    assert parts.authority == None
    assert parts.address == '/pyre/system/pyre.pml'
    assert parts.query == None
    assert parts.fragment == None

    # the full set
    parts = uri.coerce("file:///pyre.pml#anchor")
    assert parts.scheme == 'file'
    assert parts.authority == ''
    assert parts.address == '/pyre.pml'
    assert parts.query == None
    assert parts.fragment == 'anchor'

    # a poorly formed one
    try:
        uri.coerce("&")
        assert False
    except uri.CastingError as error:
        assert str(error) == "could not coerce '&' into a URI"

    # anything else?
    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
