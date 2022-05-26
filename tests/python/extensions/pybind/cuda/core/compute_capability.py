#!/usr/bin/env python3

from isce3.ext.isce3.cuda.core import ComputeCapability, min_compute_capability

def test_init():
    compute = ComputeCapability(2, 0)
    assert(compute.major == 2)
    assert(compute.minor == 0)

def test_str():
    compute = ComputeCapability(3, 5)
    assert(str(compute) == "3.5")

def test_comparison():
    compute1 = ComputeCapability(3, 2)
    compute2 = ComputeCapability(3, 2)
    compute3 = ComputeCapability(3, 5)
    compute4 = ComputeCapability(5, 0)

    assert(compute1 == compute2)
    assert(compute1 != compute3)
    assert(compute1 < compute4)
    assert(compute3 > compute1)
    assert(compute2 <= compute1)
    assert(compute4 >= compute3)

def test_min_compute():
    min_compute = min_compute_capability()
    assert(min_compute.major > 1)
    assert(min_compute.minor > 0)
