import isce3.ext.isce3 as isce3
import numpy as np
import numpy.testing as npt
from pyproj import Proj, CRS, Transformer

def test_ecef_coords():
    x0, y0 = 402538, 3851590
    dx, dy = 1, -1
    m, n = 5, 3
    epsg = 32611

    grid = isce3.product.GeoGridParameters(x0, y0, dx, dy, n, m, epsg)

    dem = isce3.geometry.DEMInterpolator()
    href = 1000.
    dem.ref_height = href

    xyz = isce3.geogrid.get_geogrid_ecef_coords(grid, dem)

    assert xyz.shape == (m, n, 3)
    assert xyz.dtype == np.float64

    # check against pyproj transform
    xform = Transformer.from_crs(
        CRS.from_epsg(epsg).to_3d(),
        CRS.from_epsg(4978).to_3d())

    xyz_ref = np.zeros_like(xyz)
    for i in range(m):
        v = y0 + dy * i
        for j in range(n):
            u = x0 + dx * j
            xyz_ref[i, j, :] = xform.transform(u, v, href)
            if not np.allclose(xyz_ref[i, j], xyz[i, j]):
                print(f"mismatch xyz[{i},{j}]:")
                for k in range(3):
                    print(f"  {xyz[i,j,k]:.3f} vs {xyz_ref[i,j,k]:.3f}")

    npt.assert_allclose(xyz, xyz_ref)
