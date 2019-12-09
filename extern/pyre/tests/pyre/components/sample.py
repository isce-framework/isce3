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

# FOR: component_class_binding_implicit.py
# declare a component that implements the job interface
class relax(pyre.component):
    """an implementation"""
    @pyre.export
    def do(self):
        """do nothing"""


# end of file
