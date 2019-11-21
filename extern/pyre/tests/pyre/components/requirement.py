#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Verify that the Requirement metaclass decorates class records properly
"""


def test():
    # access
    from pyre.components.Configurable import Configurable
    from pyre.components.Requirement import Requirement

    # declare a class
    class base(Configurable, metaclass=Requirement):
        """test class"""

    # did my ancestor list get built properly
    assert base.pyre_pedigree == (base,)

    return base


# main
if __name__ == "__main__":
    test()


# end of file
