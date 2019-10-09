#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# get access to the random number generator functions
import random
# sample size
N = 10**6
# initialize the interior point counter
interior = 0
# integrate by sampling some number of times
for i in range(N):
    # build a random point
    x = random.random()
    y = random.random()
    # check whether it is inside the unit quarter circle
    if (x*x + y*y) <= 1.0: # no need to waste time computing the sqrt
        # update the interior point counter
        interior += 1
# print the result:
print("π: {:.8f}".format(4*interior/N))


# end of file
