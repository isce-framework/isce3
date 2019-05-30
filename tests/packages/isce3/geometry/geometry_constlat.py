#!/usr/bin/env python3

import numpy as np
import isce3

def setupOrbit(lat0, lon0, omega, Nvec, ellipsoid):

    #Satellite height
    hsat = 700000.0

    # Setup orbit
    orb = isce3.core.orbit()
    t0 = isce3.core.dateTime(inobj="2017-02-12T01:12:30.0")
    orb.refEpoch = t0
    clat = np.cos(lat0)
    slat = np.sin(lat0)
    sath = ellipsoid.a + hsat
    for ii in range(Nvec):
        deltat = ii*10.0
        lon = lon0 + omega*deltat

        pos = [sath * clat * np.cos(lon),
                sath * clat * np.sin(lon),
                sath * slat]

        vel = [-omega*pos[1],
                omega*pos[0],
                0.0]

        orb.addStateVector(deltat, pos, vel)

    return orb, hsat

# Solve for Geocentric latitude given a slant range
# And look side, assuming omega is +ve
def solve(R, side, hsat, satlat0, satomega, ellipsoid):
    temp = 1 + hsat/ellipsoid.a
    temp1 = R/ellipsoid.a
    temp2 = R/(ellipsoid.a + hsat)

    cosang = 0.5 * (temp + (1.0/temp) - temp1 * temp2)
    angdiff = np.arccos(cosang);

    if ( (side * satomega) > 0):
        x = satlat0 + angdiff
    else:
        x = satlat0 - angdiff

    return x

def test_rdr2geo():
    degrees = 180.0/np.pi
    
    lon0 = 0.0
    omega = 0.1/degrees
    Nvec = 10
    lat0 = 45.0/degrees
    sides = [-1, 1]

    ellipsoid = isce3.core.ellipsoid()
    orb, hsat = setupOrbit(lat0, lon0, omega, Nvec, ellipsoid)

    for ii in range(20):
        # Azimuth time
        tinp = 5.0 + ii * 2.0

        # Slant range
        rng = 800000.0 + 10.0 * ii

        # Theoretical Solutions
        expectedLon = lon0 + omega * tinp

        for kk in range(2):
            # expected solution for geocentric latitude
            geocentricLat = solve(rng, sides[kk], hsat, lat0, omega, ellipsoid);

            # Convert geocentric coords to xyz
            xyz = [ellipsoid.a * np.cos(geocentricLat) * np.cos(expectedLon),
                   ellipsoid.a * np.cos(geocentricLat) * np.sin(expectedLon),
                   ellipsoid.a * np.sin(geocentricLat)]

            # Convert xyz to geodetic coords
            expLLH = ellipsoid.xyzToLonLat(xyz)

            # Run rdr2geo to estimate target llh
            targetLLH = isce3.geometry.radar2geoCoordinates(
                            azimuthTime=tinp, slantRange=rng, 
                            ellipsoid=ellipsoid, orbit=orb, side=sides[kk]
                            )

            print(expLLH[0] - targetLLH[0])
            print(expLLH[1] - targetLLH[1])
            print(expLLH[2] - targetLLH[2])
            assert abs(expLLH[0] - targetLLH[0]) < 1.0e-8
            assert abs(expLLH[1] - targetLLH[1]) < 1.0e-8
            assert abs(expLLH[2] - targetLLH[2]) < 1.0e-3

    return 

def test_geo2rdr():
    degrees = 180.0/np.pi

    lon0 = 0.0
    omega = 0.1/degrees
    Nvec = 10
    lat0 = 45.0/degrees
    sides = [-1, 1]

    ellipsoid = isce3.core.ellipsoid()
    orb, hsat = setupOrbit(lat0, lon0, omega, Nvec, ellipsoid)

    zeroDop = isce3.core.lut2d()

    #Dummy wavelength
    wavelength = 0.24;

    for ii in range(20):
        tinp = 25.0 + ii * 2.0
        
        for kk in range(2):

            # Determine sign
            if (omega * sides[kk]) < 0:
                sgn = 1
            else:
                sgn = -1

            # Start with geocentric lat
            geocentricLat = (lat0 + sgn * ii * 0.1/degrees)

            # Theoretical solutions
            expectedLon = lon0 + omega * tinp

            targ_xyz = [ellipsoid.a * np.cos(geocentricLat) * np.cos(expectedLon),
                        ellipsoid.a * np.cos(geocentricLat) * np.sin(expectedLon),
                        ellipsoid.a * np.sin(geocentricLat)]

            # transform to geodetic LLH
            targ_LLH = ellipsoid.xyzToLonLat(targ_xyz)

            # Expected satellite position
            sat_xyz = [(ellipsoid.a + hsat) * np.cos(lat0) * np.cos(expectedLon),
                       (ellipsoid.a + hsat) * np.cos(lat0) * np.sin(expectedLon),
                       (ellipsoid.a + hsat) * np.sin(lat0)]
            
            # line of sight vector
            los = np.array(sat_xyz) - np.array(targ_xyz)

            # expected Range 
            expRange = np.sqrt(np.sum(los**2))

            azTime, slantRange = isce3.geometry.geo2radarCoordinates(
                    lonlatheight=list(targ_LLH), 
                    ellipsoid=ellipsoid, orbit=orb, 
                    doppler=zeroDop, wavelength=0.24, 
                    threshold=1.0e-9, maxiter=50, dR=10.0
                    )

            assert abs(azTime - tinp) < 1.0e-5
            assert abs(slantRange - expRange) < 1e-8 
            
    # all done
    return

if __name__ == '__main__':
    #test_rdr2geo()
    test_geo2rdr()


