#!/usr/bin/env python3
import copy
from isce3.core import TimeDelta
import numpy.testing as npt

def load_h5():
    from isce3.core import load_orbit_from_h5_group
    from iscetest import data
    from os import path
    import h5py
    f = h5py.File(path.join(data, "envisat.h5"), 'r')
    return load_orbit_from_h5_group(f["/science/LSAR/SLC/metadata/orbit"])

o = load_h5();

def test_save():
    import h5py
    import tempfile
    o = load_h5()
    _, name = tempfile.mkstemp()
    with h5py.File(name, "w") as h5:
        g = h5.create_group("/orbit")
        o.save_to_h5(g)

# Test that accessors exist
def test_props():
    o = load_h5();
    o.time
    o.position
    o.velocity

# Test that loaded data is valid
def test_members():
    import numpy
    from numpy.linalg import norm
    o = load_h5()

    # Verify orbit type is "POE"
    # (contents of "//science/LSAR/SLC/metadata/orbit/orbitType")
    assert (o.get_type() == 'POE')

    # Check valid earth orbit distance
    earth_radius  =  6_000e3 # meters
    geostationary = 35_000e3 # meters
    altitude = numpy.array([norm(pos) for pos in o.position])
    assert(all(altitude > earth_radius))
    assert(all(altitude < geostationary))

    # Check valid orbital velocity
    # How fast should a satellite move?
    # Probably faster than a car, but slower than the speed of light.
    car = 100 * 1000./3600. # km/h to m/s
    light = 3e8
    velocity = numpy.array([norm(vel) for vel in o.velocity])
    assert(all(velocity > car))
    assert(all(velocity < light))

def test_update_epoch():
    orbit = load_h5()
    i = -1
    old_epoch = orbit.reference_epoch
    old_timestamp = old_epoch + TimeDelta(orbit.time[i])

    new_epoch = old_epoch + TimeDelta(100.0)
    orbit.update_reference_epoch(new_epoch)
    assert orbit.reference_epoch == new_epoch

    new_timestamp = orbit.reference_epoch + TimeDelta(orbit.time[i])
    assert (new_timestamp - old_timestamp).total_seconds() < 1e-9

def test_copy():
    orbit = load_h5()
    # only modifiable attribute via python is epoch
    epoch = orbit.reference_epoch + TimeDelta(1.0)
    for o in (copy.copy(orbit), copy.deepcopy(orbit), orbit.copy()):
        o.update_reference_epoch(epoch)
        assert o.reference_epoch != orbit.reference_epoch

def test_contains():
    orbit = load_h5()
    assert not orbit.contains(orbit.start_time - 1.0)
    assert not orbit.contains(orbit.end_time + 1.0)
    mid = 0.5 * (orbit.start_time + orbit.end_time)
    assert orbit.contains(mid)

def test_crop():
    orbit = load_h5()
    # want to nudge a bit past intrinsic spacing, so make sure test
    # dataset hasn't been updated
    assert orbit.time.spacing == 60.

    # a point just after the second StateVector
    start = orbit.start_datetime + TimeDelta(61)
    # a point just before the next-to-last StateVector
    stop = orbit.end_datetime - TimeDelta(61)
    cropped_orbit = orbit.crop(start, stop)

    t0 = (start - cropped_orbit.reference_epoch).total_seconds()
    t1 = (stop - cropped_orbit.reference_epoch).total_seconds()
    npt.assert_equal(cropped_orbit.size, orbit.size - 2)
    npt.assert_(cropped_orbit.contains(t0))
    npt.assert_(cropped_orbit.contains(t1))

    # Now crop around a single point with padding and make sure we get the
    # desired amount.
    start = stop = orbit.mid_datetime
    npad = 3
    cropped_orbit = orbit.crop(start, stop, npad)
    npt.assert_(cropped_orbit.size >= 2 * npad)

    # Make sure reference epoch doesn't change interp and epoch properties.
    npt.assert_(cropped_orbit.get_interp_method() == orbit.get_interp_method())
    npt.assert_(cropped_orbit.reference_epoch == orbit.reference_epoch)
