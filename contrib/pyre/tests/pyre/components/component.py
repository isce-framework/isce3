#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Check that declarations of trivial components produce the expected layout
"""


def test():
    import pyre

    # declare
    class component(pyre.component):
        """a trivial component"""

    # check the basics
    assert component.__name__ == "component"
    assert component.__bases__ == (pyre.component,)

    # check the layout
    assert component.pyre_internal is False
    assert component.pyre_pedigree == (component, pyre.component)
    assert component.pyre_namemap == {}
    assert component.pyre_traitmap == {}
    assert component.pyre_localTraits == ()
    assert component.pyre_inheritedTraits == ()
    assert component.pyre_implements == None

    assert component.pyre_inventory.key is None

    # exercise the configurable interface
    assert tuple(component.pyre_traits()) == ()
    assert component.pyre_isCompatible(component)

    # exercise the component interface
    assert component.pyre_family() is None
    assert component.pyre_package() is None
    assert tuple(component.pyre_getExtent()) == ()

    return component


# main
if __name__ == "__main__":
    test()


# end of file
