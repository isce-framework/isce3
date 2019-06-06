#!/usr/bin/env python3
import isce3
import numpy as np
a = 6378137.0
b = np.sqrt(1.0 - 0.0066943799901) * a

data = [
        ("Origin", [0.,0.,0.], [a,0.,0.]),
        ("Equator90E", [0.5*np.pi, 0., 0.], [0.,a,0.]),
        ("Equator90W",[-0.5*np.pi,0.,0.], [0.,-a,0.]),
        ("EquatorDateline", [np.pi,0.,0.], [-a,0.,0.]),
        ("NorthPole", [0.,0.5*np.pi,0.], [0.,0.,b]),
        ("SouthPole", [0.,-0.5*np.pi,0.], [0.,0.,-b]),
        ("Point1", [1.134431523585921e+00,-1.180097204507889e+00,7.552767636707697e+03],
        [1030784.925758840050548,2210337.910070449113846,-5881839.839890958741307]),
        ("Point2", [-1.988929481271171e+00,-3.218156967477281e-01,4.803829875484664e+02],
        [-2457926.302319798618555,-5531693.075449729338288,-2004656.608288598246872]),
        ("Point3", [ 3.494775870065641e-01,1.321028021250511e+00, 6.684702668405185e+03],
        [1487474.649522442836314,542090.182021118933335, 6164710.02066358923912]),
        ("Point4", [ 1.157071150199438e+00,1.539241336260909e+00,  2.075539115269004e+03],
        [81196.748833858233411,   184930.081202651723288, 6355641.007061666809022]),
        ("Point5", [ 2.903217190227029e+00,3.078348660646868e-02, 1.303664510818545e+03],
          [-6196130.955770593136549,  1505632.319945097202435,195036.854449656093493]),
        ("Point6", [ 1.404003364812063e+00,9.844570757478284e-01, 1.242074588639294e+03],
        [587386.746772550744936,  3488933.817566382698715, 5290575.784156281501055]),
        ("Point7", [1.786087533202875e+00,-1.404475795144668e+00,  3.047509859826395e+03],
        [-226426.343401445570635,  1035421.647801387240179, -6271459.446578867733479]),
        ("Point8", [ -1.535570572315143e+00,-1.394372375292064e+00, 2.520818495701064e+01],
        [39553.214744714961853, -1122384.858932408038527, -6257455.705907705239952]),
        ("Point9", [ 2.002720719284312e+00,-6.059309705813630e-01, -7.671870434220574e+01],
        [-2197035.039946643635631,  4766296.481927301734686, -3612087.398071805480868]),
        ("Point10", [ -2.340221964131008e-01,1.162119493774084e+00,  6.948177664180818e+03],
         [2475217.167525716125965,  -590067.244431337225251, 5836531.74855871964246 ]),
        ("Point11", [6.067080997777370e-01,-9.030342054807169e-01, 4.244471400804430e+02],
        [3251592.655810729600489,  2256703.30570419318974 ,-4985277.930962197482586]),
        ("Point12", [ -2.118133740176279e+00,9.812354487540356e-01, 2.921301812478523e+03],
         [-1850635.103680874686688, -3036577.247930331621319,5280569.380736761726439]),
        ("Point13", [ -2.005023821660764e+00,1.535487121535718e+00, 2.182275729585851e+02],
         [ -95048.576977927994449,  -204957.529435861855745, 6352981.530775795690715]),
        ("Point14", [2.719747828172381e+00,-1.552548149921413e+00,  4.298201230045657e+03],
        [-106608.855637043248862,    47844.679874961388123, -6359984.3118050172925]),
        ("Point15", [ -1.498660315787147e+00,1.076512019764726e+00, 8.472554905622580e+02],
         [218676.696484291809611, -3026189.824885316658765, 5592409.664520519785583])]


def test_CythonInterface():
    import numpy.testing as npt

    elp = isce3.core.ellipsoid(a=6387137., e2=0.00000001)

    assert elp.a == 6387137.
    assert elp.e2 == 0.00000001

    return

def test_lonLatToXyzPoints():
    import numpy.testing as npt

    wgs84 = isce3.core.ellipsoid()

    for entry in data:
        tname = entry[0]
        llh = entry[1]
        xyz = entry[2]

        res = wgs84.lonLatToXyz(llh)
        npt.assert_array_almost_equal(xyz, res, decimal=6, err_msg="Failed {0}".format(tname))

    return

def test_xyzToLonLatPoints():
    import numpy.testing as npt

    wgs84 = isce3.core.ellipsoid()

    for entry in data:
        tname = entry[0]
        llh = entry[1]
        xyz = entry[2]

        res = wgs84.xyzToLonLat(xyz)
        npt.assert_array_almost_equal(llh[0:2], res[0:2], decimal=9, err_msg="Failed {0}".format(tname))
        npt.assert_array_almost_equal(llh[2], res[2], decimal=6, err_msg="Failed {0}".format(tname))

    return

def test_lonLattoXyzArray():
    import numpy.testing as npt

    tname = "lon lat to xyz array"
    wgs84 = isce3.core.ellipsoid()

    nPts = len(data)

    llh = np.array([x[1] for x in data])
    xyz = np.array([x[2] for x in data])

    res = wgs84.lonLatToXyz(llh)
    npt.assert_array_almost_equal(xyz, res, decimal=6, err_msg="Failed {0}".format(tname))

def test_xyztoLonLatArray():
    from isce3.core.Ellipsoid import Ellipsoid
    import numpy.testing as npt

    tname = "xyz to lat lon array"
    wgs84 = Ellipsoid()

    nPts = len(data)

    llh = np.array([x[1] for x in data])
    xyz = np.array([x[2] for x in data])

    res = wgs84.xyzToLonLat(xyz)
    npt.assert_array_almost_equal(llh[:,:2], res[:,:2], decimal=9, err_msg="Failed {0}".format(tname))
    npt.assert_array_almost_equal(llh[:,2], res[:,2], decimal=6, err_msg="Failed {0}".format(tname))

    return

if __name__ == '__main__':

    test_CythonInterface()
    test_lonLattoXyzArray()
    test_xyztoLonLatArray()
    test_xyzToLonLatPoints()
    test_lonLatToXyzPoints()

