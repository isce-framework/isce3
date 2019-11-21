#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Exercise the path primitive
"""


def test():
    # the home of the factory
    import pyre.primitives
    # the location of this test
    here = pyre.primitives.path('/Users/mga/dv/pyre-1.0/tests/pyre/primitives/path_parts.py')

    # check that we extract the parts correctly
    assert list(here.parts) == ['/'] + str(here).split('/')[1:]

    # check that we can identify rooted paths
    assert here.anchor == '/'
    assert here.anchor == '/'

    # check that we build the sequence of parents correctly
    assert tuple(map(str, here.parents)) == (
        '/Users/mga/dv/pyre-1.0/tests/pyre/primitives',
        '/Users/mga/dv/pyre-1.0/tests/pyre',
        '/Users/mga/dv/pyre-1.0/tests',
        '/Users/mga/dv/pyre-1.0',
        '/Users/mga/dv',
        '/Users/mga',
        '/Users',
        '/')

    # check that we compute the immediate parent
    assert str(here.parent) == '/Users/mga/dv/pyre-1.0/tests/pyre/primitives'

    # verify that the {path} property returns something identical to the str representation
    assert here.path == str(here)
    # check the name
    assert here.name == 'path_parts.py'
    # check the suffix
    assert here.suffix == '.py'
    # check the stem
    assert here.stem == 'path_parts'

    # check that the path is absolute
    assert here.isAbsolute()
    # but that the cwd directory representation
    cwd = pyre.primitives.path()
    # isn't
    assert cwd.isAbsolute() is False

    # the former can be expressed as a URI
    assert here.as_uri() == 'file:///Users/mga/dv/pyre-1.0/tests/pyre/primitives/path_parts.py'
    # but the latter
    try:
        # can't
        cwd.as_uri()
        # so this is unreachable
        assert False
    # and when it fails
    except ValueError as error:
        # check that if identified the problem correctly
        assert str(error) == "'.' is a relative path and can't be expressed as a URI"

    # part replacements
    # the name
    assert str(here.withName('path_arithmetic.py')) == (
        '/Users/mga/dv/pyre-1.0/tests/pyre/primitives/path_arithmetic.py'
        )
    # replacement with an invalid name
    try:
        # which should fail
        here.withName('foo/bar')
        # so we can't get here
        assert False
    # and when it fails
    except ValueError as error:
        # check that it identified the problem correctly
        assert str(error) == "invalid name 'foo/bar'"

    # the suffix
    assert str(here.withSuffix('.pyc')) == (
        '/Users/mga/dv/pyre-1.0/tests/pyre/primitives/path_parts.pyc'
        )
    # replacement with an invalid suffix
    try:
        # which should fail
        here.withSuffix('foo')
        # so we can't get here
        assert False
    # and when it fails
    except ValueError as error:
        # check that it identified the problem correctly
        assert str(error) == "invalid suffix 'foo'"

    # all done
    return


# main
if __name__ == "__main__":
    # skip pyre initialization since we don't rely on the executive
    pyre_noboot = True
    # do...
    test()


# end of file
