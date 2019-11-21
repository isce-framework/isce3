#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Sample file with component declarations for the tests in this directory
"""


import pyre

# FOR: resolve.py
# declare a worker
class worker(pyre.component):
    """a worker"""

    @pyre.export
    def do(self):
        """do nothing"""


# end of file
