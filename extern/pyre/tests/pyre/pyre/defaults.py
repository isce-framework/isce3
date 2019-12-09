#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# get the framework
import pyre


# an app with lists
class lister(pyre.application, family='defaults.lister'):

    # the default value as {None}
    none = pyre.properties.strings(default=None)
    # the default value as a list of strings that should be expanded by the framework
    explicit = pyre.properties.strings(default=['{people.alec}'])
    # the default value as a string that evaluates to a list
    implicit = pyre.properties.strings(default='[{people.alec}]')

# an app with dicts
class dicter(pyre.component, family='str.dict'):

    # the default value as {None}
    none = pyre.properties.catalog(
        schema=pyre.properties.str(), default=None)
    # the default value as a list of strings that should be expanded by the framework
    explicit = pyre.properties.catalog(
        schema=pyre.properties.str(), default={'name': '{people.alec}'})


# the test
def test():
    # show me
    # print(lister.none)
    # check that the class defaults get evaluated correctly
    assert lister.none is None

    # show me
    # print(lister.explicit)
    # print(lister.implicit)
    # check that the class defaults get evaluated correctly
    assert lister.explicit == lister.implicit

    # instantiate
    l = lister(name='lister')
    # show me
    # print(app.explicit)
    # print(app.implicit)
    # check again
    assert l.explicit == l.implicit

    # now the dict
    # show me
    # print(dicter.none)
    # check
    assert dicter.none == None

    # show me
    # print(dict(dicter.explicit))
    # check
    assert dicter.explicit == {'name': ['alec aivazis']}

    # instantiate
    d = dicter(name='dicter')
    # show me
    # print(dict(d.explicit))
    # check
    assert d.explicit == {'name': ['alec aivazis']}

    # all done
    return l, d


# main
if __name__ == '__main__':
    test()


# end of file
