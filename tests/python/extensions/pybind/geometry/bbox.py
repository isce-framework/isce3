import iscetest
import pathlib
import isce3.ext.isce3 as isce
from isce3.core import load_orbit_from_h5_group
from nisar.products.readers import SLC


path_slc = str(pathlib.Path(iscetest.data) / "envisat.h5")


def load_orbit():
    import h5py
    with h5py.File(path_slc, "r") as h5:
        g = h5["/science/LSAR/SLC/metadata/orbit"]
        orbit = load_orbit_from_h5_group(g)
    return orbit


def test_perimeter():
    grid = isce.product.RadarGridParameters(path_slc)
    orbit = load_orbit()
    dop = isce.core.LUT2d()
    dem = isce.geometry.DEMInterpolator(0.0)
    wkt = isce.geometry.get_geo_perimeter_wkt(grid, orbit, dop, dem)
    print(wkt)
    assert wkt.startswith("POLYGON")


def test_radar_bbox():
    # define geogrid
    geogrid = isce.product.GeoGridParameters(start_x=-115.65,
                                             start_y=34.84,
                                             spacing_x=0.0002,
                                             spacing_y=-8.0e-5,
                                             width=500,
                                             length=500,
                                             epsg=4326)

    # get radar grid
    radar_grid = isce.product.RadarGridParameters(path_slc)

    # create DEM interpolator
    f_dem = f'{iscetest.data}/srtm_cropped.tif'
    dem_raster = isce.io.Raster(f_dem)
    dem_interp = isce.geometry.dem_raster_to_interpolator(dem_raster, geogrid)
    dem_interp.compute_min_max_mean_height()

    # get radar bounding box for current radar grid
    rdr_bbox = isce.geometry.get_radar_bbox(geogrid, radar_grid,
                                            load_orbit(), dem_interp)

    # lambda to test if radar bounding box indices within bounds
    in_rdr_grid_bounds = lambda ind, max_ind: ind >= 0 and ind < max_ind

    assert(rdr_bbox.first_azimuth_line < rdr_bbox.last_azimuth_line)
    assert(rdr_bbox.first_range_sample < rdr_bbox.last_range_sample)
    assert(in_rdr_grid_bounds(rdr_bbox.first_azimuth_line, radar_grid.length))
    assert(in_rdr_grid_bounds(rdr_bbox.last_azimuth_line, radar_grid.length))
    assert(in_rdr_grid_bounds(rdr_bbox.first_range_sample, radar_grid.width))
    assert(in_rdr_grid_bounds(rdr_bbox.last_range_sample, radar_grid.width))
