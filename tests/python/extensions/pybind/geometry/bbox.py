import iscetest
import pathlib
import isce3.ext.isce3 as isce

slc = str(pathlib.Path(iscetest.data) / "envisat.h5")


def load_orbit():
    import h5py
    with h5py.File(slc, "r") as h5:
        g = h5["/science/LSAR/SLC/metadata/orbit"]
        orbit = isce.core.Orbit.load_from_h5(g)
    return orbit


def test_perimeter():
    grid = isce.product.RadarGridParameters(slc)
    orbit = load_orbit()
    dop = isce.core.LUT2d()
    dem = isce.geometry.DEMInterpolator(0.0)
    wkt = isce.geometry.get_geo_perimeter_wkt(grid, orbit, dop, dem)
    print(wkt)
    assert wkt.startswith("POLYGON")
