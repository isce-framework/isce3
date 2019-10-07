#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the inheritance invariants are respected
"""


def test():
    import pyre

    # declare a couple of protocols
    class base(pyre.protocol):
        """the base protocol"""
        common = pyre.property()

    class derived(base):
        """the derived one"""
        extra = pyre.property()

    # check the basics
    assert base.__name__ == "base"
    assert base.__bases__ == (pyre.protocol,)
    # check the layout
    assert base.pyre_namemap == {'common': 'common'}
    assert base.pyre_pedigree == (base, pyre.protocol)
    # traits
    localNames = ['common']
    localTraits = tuple(map(base.pyre_trait, localNames))
    assert base.pyre_localTraits == localTraits
    assert base.pyre_inheritedTraits == ()
    allNames = localNames + []
    allTraits = list(map(base.pyre_trait, allNames))
    assert list(base.pyre_traits()) == allTraits

    # check the basics
    assert derived.__name__ == "derived"
    assert derived.__bases__ == (base, )
    # check the layout
    assert derived.pyre_namemap == {
        'common': 'common', 'extra': 'extra'
        }
    assert derived.pyre_pedigree == (derived, base, pyre.protocol)
    # traits
    localNames = ['extra']
    localTraits = tuple(map(derived.pyre_trait, localNames))
    assert derived.pyre_localTraits == localTraits
    inheritedNames = ['common']
    inheritedTraits = tuple(map(derived.pyre_trait, inheritedNames))
    assert derived.pyre_inheritedTraits == inheritedTraits
    allNames = localNames + ['common']
    allTraits = list(map(derived.pyre_trait, allNames))
    assert list(derived.pyre_traits()) == allTraits

    return base, derived


# main
if __name__ == "__main__":
    test()


# end of file
