import datetime
import numpy as np
import numpy.testing as npt

import isce3

class LinearOrbit(object):
    def __init__(self, initial_position, velocity):
        self._initial_position = initial_position
        self._velocity = velocity

    def position(self, t):
        x = self._initial_position[0] + self._velocity[0] * t
        y = self._initial_position[1] + self._velocity[1] * t
        z = self._initial_position[2] + self._velocity[2] * t
        return (x, y, z)

    def velocity(self, t):
        return self._velocity

class OrbitTest(object):
    def __init__(self):
        starttime = datetime.datetime(2000, 1, 1)
        self.starttime = isce3.core.dateTime(dt=starttime)
        self.spacing = 10.
        self.size = 11

        self.initial_position = (0., 0., 0.)
        self.velocity = (4000., -1000., 4500.)
        self.reforbit = LinearOrbit(self.initial_position, self.velocity)

        self.statevecs = []
        for i in range(self.size):
            t = i * self.spacing
            dt = self.starttime + isce3.core.timeDelta(dt=t)
            pos = self.reforbit.position(t)
            vel = self.reforbit.velocity(t)
            sv = isce3.core.statevector(datetime=dt, position=pos, velocity=vel)
            self.statevecs.append(sv)

        self.interp_times = [23.3, 36.7, 54.5, 89.3]

def testConstructor():
    test = OrbitTest()

    orbit = isce3.core.orbit(statevecs=test.statevecs)

    npt.assert_equal( orbit.referenceEpoch, test.statevecs[0].datetime )
    npt.assert_almost_equal( orbit.spacing, test.spacing, 13 )
    npt.assert_equal( orbit.size, test.size )

    for i in range(test.size):
        t = (test.statevecs[i].datetime - test.starttime).getTotalSeconds()
        npt.assert_almost_equal( orbit.time(i), t, 13 )
        npt.assert_array_equal( orbit.position(i), test.statevecs[i].position )
        npt.assert_array_equal( orbit.velocity(i), test.statevecs[i].velocity )

def testGetStateVectors():
    test = OrbitTest()

    orbit = isce3.core.orbit(statevecs=test.statevecs)
    statevecs = orbit.getStateVectors()

    npt.assert_equal( len(statevecs), test.size )

    for i in range(test.size):
        t1 = statevecs[i].datetime
        t2 = test.statevecs[i].datetime
        npt.assert_almost_equal( (t2 - t1).getTotalSeconds(), 0., 13 )
        npt.assert_array_equal( statevecs[i].position, test.statevecs[i].position )
        npt.assert_array_equal( statevecs[i].velocity, test.statevecs[i].velocity )

def testSetStateVectors():
    test = OrbitTest()

    orbit = isce3.core.orbit()
    orbit.referenceEpoch = test.statevecs[0].datetime
    orbit.setStateVectors(test.statevecs)

    npt.assert_equal( orbit.referenceEpoch, test.statevecs[0].datetime )
    npt.assert_almost_equal( orbit.spacing, test.spacing, 13 )
    npt.assert_equal( orbit.size, test.size )

    for i in range(test.size):
        t = (test.statevecs[i].datetime - test.starttime).getTotalSeconds()
        npt.assert_almost_equal( orbit.time(i), t, 13 )
        npt.assert_array_equal( orbit.position(i), test.statevecs[i].position )
        npt.assert_array_equal( orbit.velocity(i), test.statevecs[i].velocity )

def testInvalidStateVectors():
    test = OrbitTest()

    orbit = isce3.core.orbit(statevecs=test.statevecs)

    # two or more state vectors are required
    new_statevecs = test.statevecs[:1]
    with npt.assert_raises(ValueError):
        orbit.setStateVectors(new_statevecs)

    # state vectors must be uniformly sampled
    new_statevecs = test.statevecs
    new_statevecs[-1].datetime += isce3.core.timeDelta(dt=10.)
    with npt.assert_raises(ValueError):
        orbit.setStateVectors(new_statevecs)

def testReferenceEpoch():
    test = OrbitTest()

    orbit = isce3.core.orbit(statevecs=test.statevecs)
    new_refepoch = test.statevecs[1].datetime
    orbit.referenceEpoch = new_refepoch

    npt.assert_equal( orbit.referenceEpoch, new_refepoch )

    for i in range(test.size):
        t1 = test.statevecs[i].datetime
        t2 = orbit.referenceEpoch + isce3.core.timeDelta(dt=orbit.time(i))
        npt.assert_almost_equal( (t2 - t1).getTotalSeconds(), 0., 13 )

def testInterpMethod():
    test = OrbitTest()
    orbit = isce3.core.orbit(statevecs=test.statevecs, interp_method="Hermite")
    assert( orbit.interpMethod == "Hermite" )

    new_method = "Legendre"
    orbit.interpMethod = new_method
    assert( orbit.interpMethod == "Legendre" )

def testStartMidEndTime():
    # two state vectors separated by 1 second
    statevecs = []
    statevecs.append(isce3.core.statevector())
    statevecs.append(isce3.core.statevector())
    statevecs[0].datetime = isce3.core.dateTime(dt=datetime.datetime(2000, 1, 1, 0, 0, 0))
    statevecs[1].datetime = isce3.core.dateTime(dt=datetime.datetime(2000, 1, 1, 0, 0, 1))

    orbit = isce3.core.orbit(statevecs=statevecs)

    npt.assert_almost_equal( orbit.startTime, 0., 13 )
    npt.assert_almost_equal( orbit.midTime, 0.5, 13 )
    npt.assert_almost_equal( orbit.endTime, 1., 13 )

    # three state vectors with 1 second spacing
    statevecs.append(isce3.core.statevector())
    statevecs[2].datetime = isce3.core.dateTime(dt=datetime.datetime(2000, 1, 1, 0, 0, 2))

    orbit = isce3.core.orbit(statevecs=statevecs)

    npt.assert_almost_equal( orbit.startTime, 0., 13 )
    npt.assert_almost_equal( orbit.midTime, 1., 13 )
    npt.assert_almost_equal( orbit.endTime, 2., 13 )

def testStartMidEndDateTime():
    # two state vectors separated by 1 second
    statevecs = []
    statevecs.append(isce3.core.statevector())
    statevecs.append(isce3.core.statevector())
    statevecs[0].datetime = isce3.core.dateTime(dt=datetime.datetime(2000, 1, 1, 0, 0, 0))
    statevecs[1].datetime = isce3.core.dateTime(dt=datetime.datetime(2000, 1, 1, 0, 0, 1))

    orbit = isce3.core.orbit(statevecs=statevecs)

    tstart = statevecs[0].datetime
    tmid = tstart + isce3.core.timeDelta(dt=0.5)
    tend = statevecs[1].datetime
    npt.assert_almost_equal( (orbit.startDateTime - tstart).getTotalSeconds(), 0., 13 )
    npt.assert_almost_equal( (orbit.midDateTime - tmid).getTotalSeconds(), 0., 13 )
    npt.assert_almost_equal( (orbit.endDateTime - tend).getTotalSeconds(), 0., 13 )

    # three state vectors with 1 second spacing
    statevecs.append(isce3.core.statevector())
    statevecs[2].datetime = isce3.core.dateTime(dt=datetime.datetime(2000, 1, 1, 0, 0, 2))

    orbit = isce3.core.orbit(statevecs=statevecs)

    tstart = statevecs[0].datetime
    tmid = statevecs[1].datetime
    tend = statevecs[2].datetime
    npt.assert_almost_equal( (orbit.startDateTime - tstart).getTotalSeconds(), 0., 13 )
    npt.assert_almost_equal( (orbit.midDateTime - tmid).getTotalSeconds(), 0., 13 )
    npt.assert_almost_equal( (orbit.endDateTime - tend).getTotalSeconds(), 0., 13 )

def testInterpolate():
    test = OrbitTest()

    orbit = isce3.core.orbit(statevecs=test.statevecs, interp_method="Hermite")

    for t in test.interp_times:
        pos, vel = orbit.interpolate(t)
        refpos = test.reforbit.position(t)
        refvel = test.reforbit.velocity(t)
        npt.assert_array_almost_equal( pos, refpos, 8 )
        npt.assert_array_almost_equal( vel, refvel, 8 )

    orbit = isce3.core.orbit(statevecs=test.statevecs, interp_method="Legendre")

    for t in test.interp_times:
        pos, vel = orbit.interpolate(t)
        refpos = test.reforbit.position(t)
        refvel = test.reforbit.velocity(t)
        npt.assert_array_almost_equal( pos, refpos, 8 )
        npt.assert_array_almost_equal( vel, refvel, 8 )

def testOrbitInterpBorderMode():
    test = OrbitTest()

    orbit = isce3.core.orbit(statevecs=test.statevecs)
    border_mode = "FillNaN"

    t = orbit.endTime + 1.
    pos, vel = orbit.interpolate(t, border_mode)

    assert( np.isnan(pos[0]) and np.isnan(pos[1]) and np.isnan(pos[2]) )
    assert( np.isnan(vel[0]) and np.isnan(vel[1]) and np.isnan(vel[2]) )

def testSaveToH5():
    import h5py

    test = OrbitTest()
    orbit = isce3.core.orbit(statevecs=test.statevecs)

    with h5py.File("dummy-orbit.h5", "w") as f:
        g = f.create_group("/orbit")
        orbit.saveToH5(g)

    with h5py.File("dummy-orbit.h5", "r") as f:
        g = f["/orbit"]
        new_orbit = isce3.core.orbit().loadFromH5(g)

    assert( new_orbit == orbit )

def testComparison():
    test = OrbitTest()

    orbit1 = isce3.core.orbit(statevecs=test.statevecs)
    orbit2 = isce3.core.orbit(statevecs=test.statevecs)
    orbit3 = isce3.core.orbit()

    assert( orbit1 == orbit2 )
    assert( orbit1 != orbit3 )
