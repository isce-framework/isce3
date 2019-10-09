#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Check that components with properties have the expected layout
"""


def test():
    import pyre

    class component(pyre.component):
        """a trivial component"""
        p = pyre.property()

    # check the basics
    assert component.__name__ == "component"
    assert component.__bases__ == (pyre.component,)
    # check the layout
    assert component.pyre_family() is None
    assert component.pyre_namemap == {'p': 'p'}
    assert component.pyre_pedigree == (component, pyre.component)
    assert component.pyre_implements == None
    # traits
    localNames = ['p']
    localTraits = tuple(map(component.pyre_trait, localNames))
    assert component.pyre_localTraits == localTraits
    assert component.pyre_inheritedTraits == ()
    allNames = localNames + []
    allTraits = list(map(component.pyre_trait, allNames))
    assert list(component.pyre_traits()) == allTraits

    return component


# main
if __name__ == "__main__":
    test()


# end of file
