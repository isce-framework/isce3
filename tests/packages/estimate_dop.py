#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# paul a. rosen and eric m. gurrola
# jet propulsion laboratory
# california institute of technology
# (c) 2013 all rights reserved
#

"""
Verify that extensions/isce/estimate_dop.cc can be used from the mroipac shared
object to manage and run the Fortran estimate_dop.f90.
"""

import os
import sys

class EstimateDoptest(object):
    def __init__(self):
        self.rawFile = ""
        self.num_range = 0
        self.width = 0
        self.firstLine = 0
        self.lastLine = 0
        self.Ioffset = 0.
        self.Qoffset = 0.
        self.Header = 0
        self.dim1_rngDoppler = 0
        self.reltol = 1.e-12
        return

    def test(self):
        # access the estimate_dop extension
        from spisce.isce.extensions.isce import estimate_dop

        #create a rawImage and rawAccessor for the input raw data file
        import isce
        print("isce.__file__ = {}".format(isce.__file__))
        import isce.components.isceobj
        rawImage = isce.components.isceobj.createRawImage()
        rawImage.initImage(self.rawFile, 'read', self.width)
        rawImage.createImage()
        rawAccessor = rawImage.getImagePointer()

        #call the extension estimate_dop
        dopvec, fd = estimate_dop(self, rawAccessor)

        #finalize the rawImage to release memory and close the file
        rawImage.finalizeImage()

        #delete the isce.log file
        os.remove("isce.log")

        #compare the doppler samples to the apriori known test values
        for i in range(self.lastLine-self.firstLine+1):
            print("fd          = ", fd)
            print("self.truth  = ", self.truth)
            print("dopvec      = ", dopvec)
            print("self.reltol = ", self.reltol)
            assert(abs(self.truth[i]-dopvec[i]) <
                self.reltol*abs(self.truth[i]))

# main
if __name__ == "__main__":
    cdopt = EstimateDoptest()

    #set attributes for the given test.raw
    cdopt.rawFile = 'test.raw'
    cdopt.Ioffset = 15.5
    cdopt.Qoffset = 15.5
    cdopt.lastLine = 2
    cdopt.width = 6
    cdopt.header = 2
    cdopt.firstLine = 1
#    cdopt.dim1_rngDoppler = int((cdopt.Width - cdopt.Header)/2.)
    cdopt.num_range = int((cdopt.width - cdopt.header)/2.)
    cdopt.reltol = 1.e-12

    #Set the truth value of the output of estimate_dop for the given input test.raw
    #test.raw contains the following data
    #raw = ((0,0,7,13,16,18), (0,0,17,17,10,15))
    #define (skipping the 2 header samples on each line):
    #a = (raw[0][2]-Ioffset+(raw[0][3]-Qoffset)J, (raw[0][4]-Ioffset)+(raw[0][5]-Qoffset)J)
    #  = (-8.5-2.5J, 0.5+2.5J)
    #b = (raw[1][2]-Ioffset+(raw[1][3]-Qoffset)J, (raw[1][4]-Ioffset)+(raw[1][5]-Qoffset)J)
    #  = (1.5+1.5J, -4.0+13.5J)
    #p = (a[0].conjugate()*b[0], a[1].conjugate()*b[1])
    #then,
    #dop = (atan2(p[0].imag,p[0].real)/2./pi, atan2(p[1].imag,p[1].real)/2./pi)
    #    = (-0.4205265130251857, 0.2958454509218618)
    cdopt.truth = (-0.4205265130251857, 0.2958454509218618)

    #execute the test
    cdopt.test()

# end of file
