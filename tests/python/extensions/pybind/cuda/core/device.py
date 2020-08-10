#!/usr/bin/env python3

import numpy.testing as npt
import pybind_isce3 as isce3

def test_get_device_count():
    count = isce3.cuda.core.get_device_count()
    assert(count >= 0)

def test_init():
    count = isce3.cuda.core.get_device_count()
    for d in range(count):
        device = isce3.cuda.core.Device(d)
        assert(device.id == d)

        print("Device %i" % (device.id))
        print("--------")
        print("name: %s" % (device.name))
        print("compute: %s" % (device.compute_capability))
        print("total mem (bytes): %i" % (device.total_global_mem))

        assert(device.name != "")
        assert(device.total_global_mem > 0)
        assert(device.compute_capability.major >= 1)
        assert(device.compute_capability.minor >= 0)

def test_invalid_device():
    with npt.assert_raises(ValueError):
        device = isce3.cuda.core.Device(-1)

    count = isce3.cuda.core.get_device_count()
    with npt.assert_raises(ValueError):
        device = isce3.cuda.core.Device(count)

def test_get_device():
    device = isce3.cuda.core.get_device()
    assert(device.id >= 0)

def test_set_device():
    count = isce3.cuda.core.get_device_count()
    for d in range(count):
        device = isce3.cuda.core.Device(d)
        isce3.cuda.core.set_device(d)
        assert(isce3.cuda.core.get_device().id == d)

def test_comparison():
    device1 = isce3.cuda.core.Device(0)
    device2 = isce3.cuda.core.Device(0)
    assert(device1 == device2)

    count = isce3.cuda.core.get_device_count()
    if (count > 1):
        device3 = isce3.cuda.core.Device(1)
        assert(device1 != device3)
