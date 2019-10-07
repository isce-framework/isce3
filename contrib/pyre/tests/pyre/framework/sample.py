# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

import pyre

@pyre.foundry
def factory(): pass

class base(pyre.component):
    """A trivial component"""

class d1(base):
    """A trivial component subclass"""

class d2(base):
    """A trivial component subclass"""

# end of file
