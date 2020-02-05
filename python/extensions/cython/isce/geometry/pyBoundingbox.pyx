#cython: language_level=3
#
# Author: Piyush Agram
# Copyright 2017-2020
#

from boundingbox cimport getGeoPerimeter as getPerimeter
from Shapes cimport Perimeter
from libc.stdlib cimport free

def getGeoPerimeter(pyRadarGridParameters grid,
                 pyOrbit arc,
                 pyProjection proj,
                 pyLUT2d doppler=pyLUT2d(),
                 pyDEMInterpolator dem=pyDEMInterpolator(height=0.),
                 int pointsPerEdge=11,
                 double threshold=1.0e-8,
                 int numiter=15):
    '''
    Estimate perimeter for given radar grid.
    '''

    cdef Perimeter ring = getPerimeter(deref(grid.c_radargrid),
                                     arc.c_orbit,
                                     proj.c_proj,
                                     deref(doppler.c_lut),
                                     deref(dem.c_deminterp),
                                     pointsPerEdge,
                                     threshold,
                                     numiter)
    cdef char* out = ring.exportToJson()
    result = out.decode('UTF-8')
    free(out)
    return result
    
    


