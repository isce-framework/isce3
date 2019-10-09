# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#
"""
Support for the gsl_stats interface
"""

# externals
from . import gsl


# the interface for doubles
def correlation(x, y):
    """
    Compute the Pearson correlation coefficient between two vectors
    """
    # compute and return the result
    return gsl.stats_correlation(x.data, y.data)

def covariance(x, y):
    """
    Compute the covariance of two vectors
    """
    # compute and return the result
    return gsl.stats_covariance(x.data, y.data)

# end of file
