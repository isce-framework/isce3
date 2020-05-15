#-*- coding: utf-8 -*-
# Heresh Fattahi
# Copyright 2019-

# The extensions
from .. import isceextension

class Rdr2geo(isceextension.pyTopo):
    """
    Wrapper for Topo
    """
    pass

def rdr2geo_point(azimuthTime=None,
            slantRange=None,
            ellipsoid=None,
            orbit=None,
            side=None,
            doppler = 0,
            wvl = 0.24,
            threshold = 0.05,
            maxIter = 50,
            extraIter = 50,
            demInterpolatorHeight = 0):

    """
    Wrapper for py_rdr2geo standalone function.
    """


    llh = isceextension.py_rdr2geo(
            orbit, ellipsoid,
            azimuthTime, slantRange, side,
            doppler, wvl,
            threshold, maxIter, extraIter,
            demInterpolatorHeight
            )

    return llh

def rdr2geo_cone(**kw):
    """Solve for target position given radar position, range, and cone angle.
    The cone is described by a generating axis and the complement of the angle
    to that axis (e.g., angle=0 means a plane perpendicular to the axis).  The
    vertex of the cone is at the radar position, as is the center of the range
    sphere.

    Typically `axis` is the velocity vector and `angle` is the squint angle.
    However, with this interface you can also set `axis` equal to the long
    axis of the antenna, in which case `angle` is an azimuth angle.  In this
    manner one can determine where the antenna boresight intersects the ground
    at a given range and therefore determine the Doppler from pointing.

    All parameters are keyword arguments.  The following are required:
        position        Position of antenna phase center, meters ECEF XYZ.
        axis            Cone generating axis (typically velocity), ECEF XYZ.
        slantRange      Range to target, meters.
        side            Radar look direction, "left" or "right".

    These are optional:
        angle           Complement of cone angle, radians, default=0 (plane).
        demInterp       Digital elevation model, meters above ellipsoid,
                        type=pyDEMInterpolator, default=ellipsoid surface.
        threshold       Range convergence threshold, meters, default=0.05.
        maxIter         Maximum iterations, default=50.
        extraIter       Additional iterations, default=50.

    Returns ECEF [X,Y,Z] of target in meters.
    """
    return isceextension.py_rdr2geo_cone(**kw)
