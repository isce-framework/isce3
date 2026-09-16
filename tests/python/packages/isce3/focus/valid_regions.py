import h5py
import isce3
from isce3.core import LookSide
import iscetest
import math
import numpy as np
import numpy.testing as npt
from pathlib import Path
import pickle

def test_valid_regions():
    # Load arguments and expected result from HDF5 file.
    h5name = str(Path(iscetest.data) / "focus" / "sub_swaths.h5")
    with h5py.File(h5name) as h5:
        orbit = isce3.core.Orbit.load_from_h5(h5["orbit"])
        doppler = isce3.core.LUT2d.load_from_h5(h5["doppler"], "doppler")

        g = h5["rdr2geo_params"]
        rdr2geo_params = {key: dset[()] for key, dset in g.items()}

        g = h5["geo2rdr_params"]
        geo2rdr_params = {key: dset[()] for key, dset in g.items()}

        g = h5["grid"]
        side_str = g["lookside"][()].decode("utf-8").lower()
        side = LookSide.Right if side_str == "right" else LookSide.Left

        grid = isce3.product.RadarGridParameters(
            g["sensing_start"][()],
            g["wavelength"][()],
            g["prf"][()],
            g["starting_range"][()],
            g["range_pixel_spacing"][()],
            side,
            g["length"][()],
            g["width"][()],
            isce3.core.DateTime(g["ref_epoch"][()].decode("utf-8")))

        chirp_durations = h5["chirp_durations"][:]
        azres = h5["azres"][()]
        raw_bbox_lists = pickle.loads(h5["raw_bbox_lists_pickle"][()].tobytes())
        expected = h5["mask"][:]

    # Expected result was multilooked.
    looks_azimuth = math.floor(grid.length / expected.shape[0])
    looks_range = math.floor(grid.width / expected.shape[1])

    dem = isce3.geometry.DEMInterpolator()

    # Compute valid sample arrays.
    swaths = isce3.focus.get_focused_sub_swaths(raw_bbox_lists,
        chirp_durations, orbit, doppler, azres, grid, dem=dem,
        rdr2geo_params=rdr2geo_params, geo2rdr_params=geo2rdr_params)

    # Convert to boolean mask at desired posting.
    mask = np.zeros(expected.shape, dtype=bool)
    for i in range(mask.shape[0]):
        row = i * looks_azimuth
        for swath in swaths:
            start, end = np.round(swath[row] / looks_range).astype(int)
            mask[i, start:end] = True

    # Note that expected mask isn't just a result from a previous run, but
    # rather it was derived independently by other means (see README.md).
    # This means we don't expect an exact match.
    matching_fraction = np.sum(mask == expected) / np.prod(mask.shape)
    npt.assert_(matching_fraction > 0.99)

    # Same check, but via save_valid_data_mask, which rasterizes the valid
    # data polygons directly into a full-resolution mask instead of going
    # through per-pulse start/stop indices.
    full_mask = np.zeros(grid.shape, dtype=bool)
    isce3.focus.save_valid_data_mask(raw_bbox_lists, chirp_durations, orbit,
        doppler, azres, grid, full_mask, dem=dem,
        rdr2geo_params=rdr2geo_params, geo2rdr_params=geo2rdr_params)

    mask2 = full_mask[::looks_azimuth, ::looks_range][:mask.shape[0], :mask.shape[1]]
    matching_fraction = np.sum(mask2 == expected) / np.prod(mask2.shape)
    npt.assert_(matching_fraction > 0.99)
