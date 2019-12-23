#!/usr/bin/env python3
import h5py
import matplotlib.pyplot as p
import numpy as np
from numpy.polynomial.polynomial import polyval  # numpy >=1.4
from scipy.optimize import fsolve, toms748
import xml.etree.ElementTree as ET
from isce3.core import dateTime, timeDelta, statevector, orbit, ellipsoid, lut2d
from isce3.geometry import rdr2geo_point, rdr2geo_cone
from isce3 import isceextension as ie

SIDE_LEFT = "left"
SIDE_RIGHT = "right"
c = 2.99792458e8 # m/s

# TSX
side = SIDE_RIGHT
fc = 9.65e9 # Hz
wvl = c/fc
print("wavelength (m) =", wvl)
h = 3.26563146835599719E+03 # sceneAverageHeight

def _get_xml_root(filename):
    '''
    Get XML root from input file.
    '''
    with open(filename, 'r') as f:
        xml_root = ET.ElementTree(file=f).getroot()
    return xml_root

def tsx_orbit(filename):
    '''
    Parse orbit out of TerraSAR-X XML.
    '''
    root = _get_xml_root(filename)
    base = root.find('platform/orbit')
    nodes = [n for n in base if n.tag == 'stateVec']
    states = []
    for node in nodes:
        get = lambda s: node.find(s).text
        t = dateTime(dt=get('timeUTC'))
        x = [float(get(s)) for s in ('posX', 'posY', 'posZ')]
        v = [float(get(s)) for s in ('velX', 'velY', 'velZ')]
        sv = statevector(datetime=t, position=x, velocity=v)
        states.append(sv)
    #return orbit(statevecs=states, interp_method="Legendre")
    return orbit(statevecs=states)


def tsx_attitude(filename, epoch=None):
    '''
    Parse attitudes out of TerraSAR-X XML.
    '''
    root = _get_xml_root(filename)
    base = root.find('platform/attitude')
    nodes = [node for node in base if node.tag == 'attitudeData']
    n = len(nodes)
    qs = np.zeros((n, 4))
    times = []
    for i, node in enumerate(nodes):
        get = lambda s: node.find(s).text
        times.append(dateTime(dt=get('timeUTC')))
        qs[i] = [float(get(s)) for s in ('q0', 'q1', 'q2', 'q3')]
    print("number of quaternions =", len(times))
    print("attitude start time =", times[0].isoformat())
    print("attitude end time =", times[-1].isoformat())
    # XXX Attitude classes don't have an epoch.
    t0 = epoch or times[0]
    dt = np.array([(t-t0).getTotalSeconds() for t in times])
    return ie.pyQuaternion(dt, qs)


def tsx_range_bounds(filename):
    root = _get_xml_root(filename)
    t0 = float(root.find('productInfo/sceneInfo/rangeTime/firstPixel').text)
    t1 = float(root.find('productInfo/sceneInfo/rangeTime/lastPixel').text)
    return c/2*t0, c/2*t1


class TsxPoly(object):
    def __init__(self, min=None, max=None, ref=None, coeff=None):
        self.min = min
        self.max = max
        self.ref = ref
        self.coeff = coeff

    @staticmethod
    def fromXML(node):
        """Return TsxPoly from ElementTree node of type dblPolynom
        """
        getfloat = lambda s: float(node.find(s).text)
        min = getfloat('validityRangeMin')
        max = getfloat('validityRangeMax')
        ref = getfloat('referencePoint')
        n = int(node.find('polynomialDegree').text) + 1
        coeff = np.zeros(n)
        for c in node.findall('coefficient'):
            i = int(c.get('exponent'))
            coeff[i] = float(c.text)
        return TsxPoly(min=min, max=max, ref=ref, coeff=coeff)

    def __call__(self, x):
        dx = x - self.ref
        return polyval(dx, self.coeff)


class DopplerEstimate(object):
    def __init__(self, node):
        self.time = dateTime(dt=node.find('timeUTC').text)
        self.measured = TsxPoly.fromXML(node.find('basebandDoppler'))
        self.geometric = TsxPoly.fromXML(node.find('geometricDoppler'))

    
def tsx_doppler_estimates(filename):
    root = _get_xml_root(filename)
    base = root.find('processing/doppler/dopplerCentroid')
    nodes = [node for node in base if node.tag == 'dopplerEstimate']
    return [DopplerEstimate(node) for node in nodes]
    

def squint(x, orbit, attitude, sin_az=0.0):
    """Find imaging time and squint (angle between LOS and velocity) for a
    target at position x."""
    def daz(t):
        xs, _ = orbit.interpolate(t) # don't need velocity
        look = x - xs
        look *= 1.0 / np.linalg.norm(look)
        R = attitude.rotmat(t)
        err = sin_az - look.dot(R[:,0])
        #print("t, err =", t, err)
        return err
    # Find imaging time.
    #t = fsolve(daz, orbit.midTime, factor=10)[0]
    t = toms748(daz, orbit.startTime, orbit.endTime)
    #print("t0 =", t)
    # Compute squint.
    xs, vs = orbit.interpolate(t)
    vhat = vs / np.linalg.norm(vs)
    look = (x-xs) / np.linalg.norm(x-xs)
    return t, np.arcsin(look.dot(vhat))


def squint2(t, r, orbit, attitude, side, sin_az=0.0, h=0.0):
    """Find squint angle given imaging time and range to target.
    """
    p, v = orbit.interpolate(t)
    R = attitude.rotmat(t)
    axis = R[:,0]
    angle = np.arcsin(sin_az)
    ell = ie.pyEllipsoid()
    dem = ie.pyDEMInterpolator(height=h)
    llh = rdr2geo_cone(axis=axis, angle=angle, slantRange=r, position=p,
                       ellipsoid=ell, demInterp=dem, side=side)
    xyz = ell.lonLatToXyz(llh)
    look = (xyz - p) / np.linalg.norm(xyz - p)
    vhat = v / np.linalg.norm(v)
    return np.arcsin(look.dot(vhat))


def squint_to_doppler(squint, wvl, vmag):
    return 2.0 / wvl * vmag * np.sin(squint)

if __name__ == '__main__':
    fn = 'TSX1_SAR__SSC_BRX2_SM_S_SRA_20110916T010020_20110916T010025.xml'
    orbit = tsx_orbit(fn)
    # XXX Attitude class doesn't have an epoch, so match it to our Orbit's.
    epoch = orbit.referenceEpoch
    attitude = tsx_attitude(fn, epoch=epoch)
    r0, r1 = tsx_range_bounds(fn)
    nr = 16
    ranges = np.linspace(r0, r1, nr)

    print("orbit.size =", orbit.size)
    print("orbit.startDateTime =", orbit.startDateTime.isoformat())
    print("orbit.endDateTime =", orbit.endDateTime.isoformat())

    nt = orbit.size
    times = np.linspace(orbit.startTime, orbit.endTime, nt)

    dop = np.zeros((nt, nr))

    for i, t in enumerate(times):
        _, v = orbit.interpolate(t)
        for j, r in enumerate(ranges):
            sq = squint2(t, r, orbit, attitude, side, h=h)
            dop[i,j] = squint_to_doppler(sq, wvl, np.linalg.norm(v))

    lut = lut2d(x=ranges, y=times, z=dop)
    print("lut =", lut)
    print("xStart =", lut.xStart)

    # dump to HDF5
    with h5py.File("out.h5", "w") as f:
        group = f.create_group("/orbit")
        orbit.saveToH5(group)
        group = f.create_group("/attitude")
        attitude.saveToH5(group)
        group = f.create_group("/doppler")
        lut.saveCalGrid(group, "dopplerCentroid", epoch, units="Hz")

    # plot doppler LUT2d
    nx = 200
    ny = 200
    x = np.linspace(r0, r1, nx)
    y = np.linspace(times[0], times[-1], ny)
    yy = np.zeros_like(x)
    z = np.zeros((ny,nx))
    for i, yi in enumerate(y):
        yy.fill(yi)
        z[i,:] = lut(yy, x)

    p.figure()
    p.pcolormesh(x/1000, y, z)
    p.xlabel("Range (km)")
    p.ylabel("Time (s)")
    p.colorbar().set_label("Hz")

    # plot comparison with annotated Doppler curves
    estimates = tsx_doppler_estimates(fn)
    dlr = estimates[len(estimates)//2]
    t = (dlr.time - epoch).getTotalSeconds()
    yy.fill(t)
    dop_pointing = lut(yy, x)
    dop_dlr_geo = np.zeros_like(dop_pointing)
    dop_dlr_meas = np.zeros_like(dop_pointing)
    for i, r in enumerate(x):
        tau = r * 2 / c  # XXX squinted range?
        dop_dlr_geo[i] = dlr.geometric(tau)
        dop_dlr_meas[i] = dlr.measured(tau)

    p.figure()
    p.plot(x/1000, dop_dlr_geo, label="DLR Geometry")
    p.plot(x/1000, dop_dlr_meas, label="DLR Measured")
    p.plot(x/1000, dop_pointing, label="LUT")
    p.xlabel("Range (km)")
    p.ylabel("Doppler (Hz)")
    p.legend()

    p.show()
