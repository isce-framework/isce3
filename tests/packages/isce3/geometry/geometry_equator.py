#!/usr/bin/env python3

import numpy as np
import isce3

def setupOrbit(lon0, omega, Nvec, ellipsoid):

    #Satellite height
    hsat = 700000.0

    # Setup orbit
    orb = isce3.core.orbit()
    t0 = isce3.core.dateTime(dt="2017-02-12T01:12:30.0")
    orb.refEpoch = t0
    for ii in range(Nvec):
        deltat = ii*10.0
        lon = lon0 + omega*deltat

        pos = [(ellipsoid.a + hsat) * np.cos(lon),
                (ellipsoid.a + hsat) * np.sin(lon),
                0.0]

        vel = [-omega*pos[1],
                omega*pos[0],
                0.0]

        orb.addStateVector(deltat, pos, vel)

    return orb, hsat

# Solve for Geocentric latitude 
def solve(R, hsat, ellipsoid):
    temp = 1 + hsat/ellipsoid.a
    temp1 = R/ellipsoid.a
    A = ellipsoid.e2
    B = -2.0 * temp
    C = temp * temp + 1.0 - ellipsoid.e2 - temp1*temp1

    # solve quadratic equation
    D = np.sqrt(B**2 - 4*A*C)

    x1 = (D-B)/(2.0*A)
    x2 = -(D+B)/(2.0*A)

    if np.abs(x2) < np.abs(x1):
        x = x2
    else:
        x = x1

    return x

def test_rdr2geo():
    degrees = 180.0/np.pi
    
    lon0 = 0.0
    omega = 0.1/degrees
    Nvec = 10

    ellipsoid = isce3.core.ellipsoid()
    orb, hsat = setupOrbit(lon0, omega, Nvec, ellipsoid)

    for ii in range(20):
        # Azimuth time
        tinp = 5.0 + ii * 2.0

        # Slant range
        rng = 800000.0 + 10.0 * ii

        # Theoretical Solutions
        expectedLon = lon0 + omega * tinp

        # expected solution for geocentric latitude
        geocentricLat = np.arccos(solve(rng, hsat, ellipsoid));

        # Convert geocentric coords to xyz
        xyz = [ellipsoid.a * np.cos(geocentricLat) * np.cos(expectedLon),
                ellipsoid.a * np.cos(geocentricLat) * np.sin(expectedLon),
                ellipsoid.b * np.sin(geocentricLat)]

        # Convert xyz to geodetic coords
        expLLH = ellipsoid.xyzToLonLat(xyz)

        # Run rdr2geo to estimate target llh
        targetLLH = isce3.geometry.radar2geo_point(
                azimuthTime=tinp, slantRange=rng, 
                ellipsoid=ellipsoid, orbit=orb, side=1.0, 
                threshold = 1e-8, maxIter = 25, extraIter = 15
                )

        assert abs(expLLH[0] - targetLLH[0]) < 1.0e-8
        assert abs(expLLH[1] - targetLLH[1]) < 1.0e-8
        assert abs(expLLH[2] - targetLLH[2]) < 1.0e-8

        # Run rdr2geo again with right looking side
        targetLLH = isce3.geometry.radar2geo_point(
                azimuthTime=tinp, slantRange=rng, 
                ellipsoid=ellipsoid, orbit=orb, side=-1.0,
                threshold = 1e-8, maxIter = 25, extraIter = 15
                )

        assert abs(expLLH[0] - targetLLH[0]) < 1.0e-8
        assert abs(expLLH[1] + targetLLH[1]) < 1.0e-8
        assert abs(expLLH[2] - targetLLH[2]) < 1.0e-8

    return

def test_geo2rdr():
    degrees = 180.0/np.pi

    lon0 = 0.0
    omega = 0.1/degrees
    Nvec = 10

    ellipsoid = isce3.core.ellipsoid()
    orb, hsat = setupOrbit(lon0, omega, Nvec, ellipsoid)

    zeroDop = isce3.core.lut2d()

    #Dummy wavelength
    wavelength = 0.24;

    for ii in range(20):
        tinp = 25.0 + ii * 2.0
        
        # Start with geocentric lat
        geocentricLat = (2.0 + ii * 0.1)/degrees

        # Theoretical solutions
        expectedLon = lon0 + omega * tinp

        targ_xyz = [ellipsoid.a * np.cos(geocentricLat) * np.cos(expectedLon),
                    ellipsoid.a * np.cos(geocentricLat) * np.sin(expectedLon),
                    ellipsoid.b * np.sin(geocentricLat)]

        # transform to geodetic LLH
        targ_LLH = ellipsoid.xyzToLonLat(targ_xyz)

        # Expected satellite position
        sat_xyz = [(ellipsoid.a + hsat) * np.cos(expectedLon),
                    (ellipsoid.a + hsat) * np.sin(expectedLon),
                    0.0]
        
        # line of sight vector
        los = np.array(sat_xyz) - np.array(targ_xyz)

        # expected Range 
        expRange = np.sqrt(np.sum(los**2))

        azTime, slantRange = isce3.geometry.geo2radar_point(
                lonlatheight=list(targ_LLH), ellipsoid=ellipsoid, 
                orbit=orb, doppler=zeroDop, 
                wavelength=0.24, threshold=1.0e-9, maxiter=50, dR=10.0
                )

        assert abs(azTime - tinp) < 1.0e-6
        assert abs(slantRange - expRange) < 1e-8 

    return

if __name__ == '__main__':
    test_rdr2geo()
    test_geo2rdr()

