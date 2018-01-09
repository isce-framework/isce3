#!/usr/bin/env python3

import numpy as np
import pyproj

def testGeocent():
    '''
    Driver to generate test data for geocentric projection.
    '''

    llh = np.random.rand(15,3)
    llh[:,0] = -np.pi + llh[:,0] * 2 * np.pi
    llh[:,1] = -0.5*np.pi + llh[:,1] * np.pi
    llh[:,2] = -500 + llh[:,2] * 9000.


    xyz = np.zeros(llh.shape)

    wgs84 = pyproj.Proj("+proj=latlon +datum=WGS84")
    ecef = pyproj.Proj("+proj=geocent +datum=WGS84")

    xx,yy,zz = pyproj.transform(wgs84,ecef,np.degrees(llh[:,0]), np.degrees(llh[:,1]), llh[:,2])

    xyz[:,0] = xx
    xyz[:,1] = yy
    xyz[:,2] = zz


    return llh, xyz


def testSouthPolar():
    '''
    Driver to generate test data for South Polar projection.
    '''

    llh = np.random.rand(15,3)
    llh[:,0] = -np.pi + llh[:,0] * 2 * np.pi
    llh[:,1] = -0.5*np.pi + llh[:,1] * np.radians(34.0)
    llh[:,2] = -500 + llh[:,2] * 3000.

    enu = np.zeros(llh.shape)

    wgs84 = pyproj.Proj(init='EPSG:4326')
    polar = pyproj.Proj(init='EPSG:3031')

    ee,nn,uu = pyproj.transform(wgs84, polar, np.degrees(llh[:,0]),
                    np.degrees(llh[:,1]), llh[:,2])

    enu[:,0] = ee
    enu[:,1] = nn
    enu[:,2] = uu

    return llh, enu

def testNorthPolar():
    '''
    Driver to generate test data for South Polar projection.
    '''

    llh = np.random.rand(15,3)
    llh[:,0] = -np.pi + llh[:,0] * 2 * np.pi
    llh[:,1] = 0.5*np.pi - llh[:,1] * np.radians(34.0)
    llh[:,2] = -500 + llh[:,2] * 3000.

    enu = np.zeros(llh.shape)

    wgs84 = pyproj.Proj(init='EPSG:4326')
    polar = pyproj.Proj(init='EPSG:3413')

    ee,nn,uu = pyproj.transform(wgs84, polar, np.degrees(llh[:,0]),
                    np.degrees(llh[:,1]), llh[:,2])

    enu[:,0] = ee
    enu[:,1] = nn
    enu[:,2] = uu

    return llh, enu



def testUTMNorth():
    '''
    Driver to generate test data for South Polar projection.
    '''

    llh = np.random.rand(60,3)
    llh[:,0] = np.radians(-177.0 + (llh[:,0] - 0.5) *6 + np.arange(60)*6)
    llh[:,1] = np.radians(80) * llh[:,1] 
    llh[:,2] = -500 + llh[:,2] * 3000.

    enu = np.zeros(llh.shape)

    wgs84 = pyproj.Proj(init='EPSG:4326')

    for zone in range(1,61):
        polar = pyproj.Proj(init='EPSG:{0}'.format(32600+zone))

        ee,nn,uu = pyproj.transform(wgs84, polar, np.degrees(llh[zone-1,0]),
                    np.degrees(llh[zone-1,1]), llh[zone-1,2])

        enu[zone-1,0] = ee
        enu[zone-1,1] = nn
        enu[zone-1,2] = uu

    return llh, enu

def testUTMSouth():
    '''
    Driver to generate test data for South Polar projection.
    '''

    llh = np.random.rand(60,3)
    llh[:,0] = np.radians(-177.0 + (llh[:,0] - 0.5) *6 + np.arange(60)*6)
    llh[:,1] = -np.radians(80) * llh[:,1] 
    llh[:,2] = -500 + llh[:,2] * 3000.

    enu = np.zeros(llh.shape)

    wgs84 = pyproj.Proj(init='EPSG:4326')

    for zone in range(1,61):
        polar = pyproj.Proj(init='EPSG:{0}'.format(32700+zone))

        ee,nn,uu = pyproj.transform(wgs84, polar, np.degrees(llh[zone-1,0]),
                    np.degrees(llh[zone-1,1]), llh[zone-1,2])

        enu[zone-1,0] = ee
        enu[zone-1,1] = nn
        enu[zone-1,2] = uu

    return llh, enu


def testCEA():
    '''
    Driver to generate test data for geocentric projection.
    '''

    llh = np.random.rand(15,3)
    llh[:,0] = -np.pi + llh[:,0] * 2 * np.pi
    llh[:,1] = -0.5*np.pi + llh[:,1] * np.pi
    llh[:,2] = -500 + llh[:,2] * 9000.


    xyz = np.zeros(llh.shape)

    wgs84 = pyproj.Proj(init="EPSG:4326")
    ecef = pyproj.Proj("+proj=cea +lon_0=0 +lat_ts=30 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

    xx,yy,zz = pyproj.transform(wgs84,ecef,np.degrees(llh[:,0]), np.degrees(llh[:,1]), llh[:,2])

    xyz[:,0] = xx
    xyz[:,1] = yy
    xyz[:,2] = zz


    return llh, xyz
