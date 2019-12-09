#!/usr/bin/env python3

import numpy as np

def test_cuimport():
    # Need to import isceextension first to allow access to core routines
    import isce3.extensions.isceextension
    # Now import iscecudaextension
    import isce3.extensions.iscecudaextension

# end of file
