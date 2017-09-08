#!/usr/bin/env python3

import numpy as np
import pyproj

if __name__ == '__main__':
    '''
    Driver to generate test data.
    '''

    llh = np.random.rand(15,3)
    llh[:,0] = -0.5*np.pi + llh[:,0] * np.pi
    llh[:,1] = -np.pi + llh[:,1] * 2 * np.pi
    llh[:,2] = -500 + llh[:,2] * 9000.


    xyz = np.zeros(llh.shape)

    wgs84 = pyproj.Proj("+proj=latlon +datum=WGS84")
    ecef = pyproj.Proj("+proj=geocent +datum=WGS84")

    xx,yy,zz = pyproj.transform(wgs84,ecef,np.degrees(llh[:,1]), np.degrees(llh[:,0]), llh[:,2])

    xyz[:,0] = xx
    xyz[:,1] = yy
    xyz[:,2] = zz
