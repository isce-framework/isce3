#!/usr/bin/env python3

import numpy as np
import isce3

class LinearOrbit():
    '''
    Make example straight line orbit.
    '''

    def __init__(self):
        self.t0 = 1000.
        self.dt = 10.
        self.nVec = 11
    
        self.opos = np.zeros(3)
        self.ovel = np.array([4000., -1000., 4500.])

    def makeStateVector(self, tdiff):
        t = self.t0 + tdiff
        pos = self.ovel * tdiff
        vel = self.ovel

        return t, pos, vel

    def makeOrbit(self):
        #from isce3.core import Orbit
        orb = isce3.core.orbit(nVectors=self.nVec)
        for ii in range(self.nVec):
            t, pos, vel = self.makeStateVector(ii*self.dt)
            orb.setStateVector(ii, t, pos, vel)

        return orb


class CirculatOrbit():
    '''
    Make example circular orbit.
    '''

    def __init__(self):
        self.t0 = 1000.
        self.dt = 10.
        self.nVec = 11

        self.omega1 = 2 * np.pi / 7000.
        self.omega2 = 2 * np.pi / 4000.
        self.theta1 = 2 * np.pi / 8.
        self.theta2 = 2 * np.pi / 12.
        self.radius = 8000000.
        self.opos = np.array([7000000., -4500000., 7800000.])


    def makeStateVector(self, tdiff):

        ang1 = self.theta1 + tdiff * self.omega1
        ang2 = self.theta2 + tdiff * self.omega2

        t = self.t0 + tdiff
        pos = self.opos +  self.radius * np.array([np.cos(ang1),
                        np.sin(ang1) + np.cos(ang2),
                        np.sin(ang2)])

        vel = self.radius * np.array([-self.omega1 * np.sin(ang1),
                        (self.omega1 * np.cos(ang1) - self.omega2 * np.sin(ang2)),
                        self.omega2 * np.cos(ang2)])

        return t, pos, vel

    def makeOrbit(self):
        #from isce3.core import Orbit
        orb = isce3.core.orbit(nVectors=self.nVec)
        for ii in range(self.nVec):
            t, pos, vel = self.makeStateVector(ii*self.dt)
            orb.setStateVector(ii, t, pos, vel)

        return orb

class PolynomialOrbit:
    '''
    Make example polynomial orbit.
    '''

    def __init__(self):
        self.t0 = 1000.
        self.dt = 10.
        self.nVec = 11

        self.xpoly = np.array([-7000000., 5435., -45.0, 7.3][::-1])
        self.ypoly = np.array([5400000., -4257., 23.0, 3.9, 0.01][::-1])
        self.zpoly = np.array([0.0, 7000., 11.0][::-1])

        self.vxpoly = np.polyder(self.xpoly)
        self.vypoly = np.polyder(self.ypoly)
        self.vzpoly = np.polyder(self.zpoly)

    def makeStateVector(self, tdiff):
        t = self.t0 + tdiff 

        pos = np.zeros(3)
        pos[0] = np.polyval(self.xpoly, tdiff)
        pos[1] = np.polyval(self.ypoly, tdiff)
        pos[2] = np.polyval(self.zpoly, tdiff)

        vel = np.zeros(3)
        vel[0] = np.polyval(self.vxpoly, tdiff)
        vel[1] = np.polyval(self.vypoly, tdiff)
        vel[2] = np.polyval(self.vzpoly, tdiff)

        return t, pos, vel

    def makeOrbit(self):
        from isce3.core.Orbit import Orbit
        orb = Orbit(nVectors=self.nVec)
        for ii in range(self.nVec):
            t, pos, vel = self.makeStateVector(ii * self.dt)
            orb.setStateVector(ii, t, pos, vel)

        return orb


linorb = LinearOrbit()
circorb = CirculatOrbit()
polyorb = PolynomialOrbit()

def test_OutOfBounds():
    orb = linorb.makeOrbit()
    test_t = [-23.0, -1.0, 101.0, 112.0]

    for tt in test_t:

        tinp = linorb.t0 + tt

        stat, pos, vel = orb.interpolate(tinp, method='hermite')
        assert (stat == 1)

        stat, pos, vel = orb.interpolate(tinp, method='sch')
        assert (stat == 1)

        stat, pos, vel = orb.interpolate(tinp, method='legendre')
        assert (stat == 1)

def test_Edges():
    import numpy.testing as npt
    
    orb = linorb.makeOrbit()
    test_t = [0., (linorb.nVec-1) * linorb.dt]
    for tt in test_t:
        tinp = linorb.t0 + tt
        tref, refpos, refvel = linorb.makeStateVector(tt)
            
        for method in ['hermite', 'sch', 'legendre']:
            stat, pos, vel = orb.interpolate(tinp, method='hermite')
            assert (stat == 0)

            npt.assert_array_almost_equal(pos, refpos, decimal=6, err_msg="Failed in edge test for {0}".format(method))

            npt.assert_array_almost_equal(vel, refvel, decimal=6, err_msg="Failed in edge test for {0}".format(method))

            
def test_Reverse():
    import numpy.testing as npt

    orb = linorb.makeOrbit()

    neworb = isce3.core.orbit()
    for ii in range(linorb.nVec-1,-1,-1):
        t, pos, vel = linorb.makeStateVector(ii * linorb.dt)
        neworb.addStateVector(t, pos, vel)

    
    for ii in range(linorb.nVec):
        t, pos, vel = neworb.getStateVector(ii)
        tref, posref, velref = orb.getStateVector(ii)

        npt.assert_array_equal(posref, pos, err_msg="Position {0}".format(ii))
        npt.assert_array_equal(velref, vel, err_msg="Velocity {0}".format(ii))


def test_OutOfOrder():
    import numpy.testing as npt

    orb = linorb.makeOrbit()

    neworb = isce3.core.orbit()
    for ii in range(linorb.nVec-1,-1,-2):
        t, pos, vel = linorb.makeStateVector(ii * linorb.dt)
        neworb.addStateVector(t, pos, vel)

    for ii in range(1,linorb.nVec,2):
        t, pos, vel = linorb.makeStateVector(ii * linorb.dt)
        neworb.addStateVector(t, pos, vel)

    
    for ii in range(linorb.nVec):
        t, pos, vel = neworb.getStateVector(ii)
        tref, posref, velref = orb.getStateVector(ii)

        npt.assert_array_equal(posref, pos, err_msg="Position {0}".format(ii))
        npt.assert_array_equal(velref, vel, err_msg="Velocity {0}".format(ii))



def test_Hermite():
    import numpy.testing as npt

    test_tt = [23.3, 36.7, 54.5, 89.3]
    for src in [linorb, circorb, polyorb]:
        orb = src.makeOrbit()
        for tt in test_tt:
            tref, refpos, refvel = src.makeStateVector(tt)

            t, pos, vel = orb.interpolate(src.t0 + tt, method='hermite')

            npt.assert_array_almost_equal(refpos, pos, decimal=5, err_msg='Position {0}'.format(tt))
            npt.assert_array_almost_equal(refvel, vel, decimal=6, err_msg='Velocity {0}'.format(tt))

            t, pos, vel = orb.interpolateWGS84Orbit(src.t0 + tt)

            npt.assert_array_almost_equal(refpos, pos, decimal=5, err_msg='Pos {0}'.format(tt))
            npt.assert_array_almost_equal(refvel, vel, decimal=6, err_msg='Vel {0}'.format(tt))


def test_SCH():
    import numpy.testing as npt

    test_tt = [23.3, 36.7, 54.5, 89.3]
    for src in [linorb, circorb, polyorb]:
        orb = src.makeOrbit()
        for tt in test_tt:
            tref, refpos, refvel = src.makeStateVector(tt)

            t, pos, vel = orb.interpolate(src.t0 + tt, method='sch')

            npt.assert_array_almost_equal(refpos, pos, decimal=5, err_msg='Position {0}'.format(tt))
            npt.assert_array_almost_equal(refvel, vel, decimal=6, err_msg='Velocity {0}'.format(tt))

            t, pos, vel = orb.interpolateSCHOrbit(src.t0 + tt)

            npt.assert_array_almost_equal(refpos, pos, decimal=5, err_msg='Pos {0}'.format(tt))
            npt.assert_array_almost_equal(refvel, vel, decimal=6, err_msg='Vel {0}'.format(tt))



def test_Legendre():
    import numpy.testing as npt

    test_tt = [23.3, 36.7, 54.5, 89.3]
    for src in [linorb, circorb, polyorb]:
        orb = src.makeOrbit()
        for tt in test_tt:
            tref, refpos, refvel = src.makeStateVector(tt)

            t, pos, vel = orb.interpolate(src.t0 + tt, method='legendre')

            npt.assert_array_almost_equal(refpos, pos, decimal=5, err_msg='Position {0}'.format(tt))
            npt.assert_array_almost_equal(refvel, vel, decimal=6, err_msg='Velocity {0}'.format(tt))

            t, pos, vel = orb.interpolateLegendreOrbit(src.t0 + tt)

            npt.assert_array_almost_equal(refpos, pos, decimal=5, err_msg='Pos {0}'.format(tt))
            npt.assert_array_almost_equal(refvel, vel, decimal=6, err_msg='Vel {0}'.format(tt))


def test_Hermite_numpy():
    import numpy.testing as npt

    test_tt = [23.3, 36.7, 54.5, 89.3]
    orb = polyorb.makeOrbit()
    
    posref = []
    velref = []
    tinp = []
    for tt in test_tt:
        t, pos, vel = orb.interpolate(polyorb.t0 + tt, method='hermite')
        posref.append(pos)
        velref.append(vel)
        tinp.append(polyorb.t0 + tt)
    
    ts, poss, vels = orb.interpolate(tinp, method='hermite')

    npt.assert_array_equal(np.array(posref), poss)
    npt.assert_array_equal(np.array(velref), vels)


    posref = []
    velref = []
    tinp = []
    for tt in test_tt:
        t, pos, vel = orb.interpolateWGS84Orbit(polyorb.t0 + tt)
        posref.append(pos)
        velref.append(vel)
        tinp.append(polyorb.t0 + tt)
    
    ts, poss, vels = orb.interpolateWGS84Orbit(tinp)

    npt.assert_array_equal(np.array(posref), poss)
    npt.assert_array_equal(np.array(velref), vels)


def test_Legendre_numpy():
    import numpy.testing as npt

    test_tt = [23.3, 36.7, 54.5, 89.3]
    orb = polyorb.makeOrbit()
    
    posref = []
    velref = []
    tinp = []
    for tt in test_tt:
        t, pos, vel = orb.interpolate(polyorb.t0 + tt, method='legendre')
        posref.append(pos)
        velref.append(vel)
        tinp.append(polyorb.t0 + tt)
    
    ts, poss, vels = orb.interpolate(tinp, method='legendre')

    npt.assert_array_equal(np.array(posref), poss)
    npt.assert_array_equal(np.array(velref), vels)


    posref = []
    velref = []
    tinp = []
    for tt in test_tt:
        t, pos, vel = orb.interpolateLegendreOrbit(polyorb.t0 + tt)
        posref.append(pos)
        velref.append(vel)
        tinp.append(polyorb.t0 + tt)
    
    ts, poss, vels = orb.interpolateLegendreOrbit(tinp)

    npt.assert_array_equal(np.array(posref), poss)
    npt.assert_array_equal(np.array(velref), vels)


def test_SCH_numpy():
    import numpy.testing as npt
    
    test_tt = [23.3, 36.7, 54.5, 89.3]
    orb = polyorb.makeOrbit()
    posref = []
    velref = []
    tinp = []
    for tt in test_tt:
        t, pos, vel = orb.interpolate(polyorb.t0 + tt, method='sch')
        posref.append(pos)
        velref.append(vel)
        tinp.append(polyorb.t0 + tt)
    
    ts, poss, vels = orb.interpolate(tinp, method='sch')

    npt.assert_array_equal(np.array(posref), poss)
    npt.assert_array_equal(np.array(velref), vels)


    posref = []
    velref = []
    tinp = []
    for tt in test_tt:
        t, pos, vel = orb.interpolateSCHOrbit(polyorb.t0 + tt)
        posref.append(pos)
        velref.append(vel)
        tinp.append(polyorb.t0 + tt)
    
    ts, poss, vels = orb.interpolateSCHOrbit(tinp)
    npt.assert_array_equal(np.array(posref), poss)
    npt.assert_array_equal(np.array(velref), vels)


if __name__ == "__main__":
    test_OutOfBounds()
    test_Edges()
    test_Reverse()
    test_OutOfOrder()
    test_Hermite()
    test_SCH()
    test_Legendre()
    test_Hermite_numpy()
    test_Legendre_numpy()
    test_SCH_numpy()
