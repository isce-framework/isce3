import h5py
import logging
import numpy as np
from numpy.testing import assert_allclose
import os
import isce3
from isce3.core import LUT2d, DateTime, Orbit, Attitude, EulerAngles
from isce3.product import RadarGridParameters
from isce3.geometry import DEMInterpolator
from nisar.h5 import set_string
from nisar.types import complex32
from nisar.products.readers.Raw import Raw
from nisar.workflows.h5_prep import add_geolocation_grid_cubes_to_hdf5

log = logging.getLogger("SLCWriter")

# TODO refactor isce::io::setRefEpoch
def time_units(epoch: DateTime) -> str:
    # XXX isce::io::*RefEpoch don't parse or serialize fraction.
    if epoch.frac != 0.0:
        raise ValueError("Reference epoch must have integer seconds.")
    date = "{:04d}-{:02d}-{:02d}".format(epoch.year, epoch.month, epoch.day)
    time = "{:02d}:{:02d}:{:02d}".format(epoch.hour, epoch.minute, epoch.second)
    return "seconds since " + date + " " + time


def assert_same_lut2d_grid(x: np.ndarray, y: np.ndarray, lut: LUT2d):
    assert_allclose(x[0], lut.x_start)
    assert_allclose(y[0], lut.y_start)
    assert (len(x) > 1) and (len(y) > 1)
    assert_allclose(x[1] - x[0], lut.x_spacing)
    assert_allclose(y[1] - y[0], lut.y_spacing)
    assert lut.width == len(x)
    assert lut.length == len(y)


def h5_require_dirname(group: h5py.Group, name: str):
    """Make sure any intermediate paths in `name` exist in `group`.
    """
    assert os.sep == '/', "Need to fix HDF5 path manipulation on Windows"
    d = os.path.dirname(name)
    group.require_group(d)


def add_cal_layer(group: h5py.Group, lut: LUT2d, name: str,
                  epoch: DateTime, units: str):
    """Add calibration LUT to HDF5 group, making sure that its domain matches
    any existing cal LUTs.

    Parameters
    ----------
    group : h5py.Group
        Group where LUT and its axes will be stored.
    lut : isce3.core.LUT2d
        Look up table.  Axes are assumed to be y=zeroDopplerTime, x=slantRange.
    name : str
        Name of dataset to store LUT data (can be path relative to `group`).
    epoch: isce3.core.DateTime
        Reference time associated with y-axis.
    units : str
        Units of lut.data.  Will be stored in `units` attribute of dataset.
    """
    # If we've already saved one cal layer then we've already saved its
    # x and y axes.  Make sure they're the same and just save the new z data.
    xname, yname = "slantRange", "zeroDopplerTime"
    if (xname in group) and (yname in group):
        x, y = group[xname], group[yname]
        assert_same_lut2d_grid(x, y, lut)
        extant_epoch = isce3.io.get_ref_epoch(y.parent, y.name)
        assert extant_epoch == epoch
        data = lut.data
        z = group.require_dataset(name, data.shape, data.dtype)
        z[...] = data
    elif (xname not in group) and (yname not in group):
        h5_require_dirname(group, name)
        lut.save_to_h5(group, name, epoch, units)
    else:
        raise IOError(f"Found only one of {xname} or {yname}."
                      "  Need both or none.")


class SLC(h5py.File):
    def __init__(self, *args, band="LSAR", product="RSLC", **kw):
        super().__init__(*args, **kw)
        self.band = band
        self.product = product
        self.root = self.create_group(f"/science/{band}/{product}")
        self.idpath = f"/science/{band}/identification"
        self.attrs["Conventions"] = np.string_("CF-1.7")
        self.attrs["contact"] = np.string_("nisarops@jpl.nasa.gov")
        self.attrs["institution"] = np.string_("NASA JPL")
        self.attrs["mission_name"] = np.string_("NISAR")
        self.attrs["reference_document"] = np.string_("TBD")
        self.attrs["title"] = np.string_("NISAR L1 RSLC Product")

    def create_dataset(self, *args, **kw):
        log.debug(f"Creating dataset {args[0]}")
        return super().create_dataset(*args, **kw)

    def create_group(self, *args, **kw):
        log.debug(f"Creating group {args[0]}")
        return super().create_group(*args, **kw)

    def set_parameters(self, dop: LUT2d, epoch: DateTime, frequency='A'):
        log.info(f"Saving Doppler for frequency {frequency}")
        g = self.root.require_group("metadata/processingInformation/parameters")
        # Actual LUT goes into a subdirectory, not created by serialization.
        name = f"frequency{frequency}"
        fg = g.require_group(name)
        add_cal_layer(g, dop, f"{name}/dopplerCentroid", epoch, "Hz")
        # TODO veff, fmrate not used anywhere afaict except product io.
        v = np.zeros_like(dop.data)
        g.require_dataset("effectiveVelocity", v.shape, v.dtype, data=v)
        fg.require_dataset("azimuthFMRate", v.shape, v.dtype, data=v)
        # TODO weighting, ref height
        if "rangeChirpWeighting" not in g:
            g.require_dataset("rangeChirpWeighting", v.shape, np.float32, 
                              data=v)
        if "referenceTerrainHeight" not in g:
            ref_terrain_height = np.zeros((v.shape[0]))
            g.require_dataset("referenceTerrainHeight", (v.shape[0],), 
                              np.float32, data=ref_terrain_height)

        # TODO populate processingInformation/algorithms
        algorithms_ds = self.root.require_group("metadata/processingInformation/algorithms")
        algorithms_dataset_list = ["ISCEVersion", 
                                   "SWSTCorrection", 
                                   "azimuthCompression", 
                                   "azimuthPresumming", 
                                   "dopplerCentroidEstimation", 
                                   "driftCompensator", 
                                   "elevationAntennaPatternCorrection", 
                                   "internalCalibration", 
                                   "patchProcessing", 
                                   "postProcessing", 
                                   "rangeCellMigration", 
                                   "rangeCompression", 
                                   "rangeDependentGainCorrection", 
                                   "rangeReferenceFunctionGenerator", 
                                   "rangeSpreadingLossCorrection", 
                                   "secondaryRangeCompression"]

        for algorithm in algorithms_dataset_list:
            if algorithm in g:
                continue
            algorithms_ds.require_dataset(algorithm, (), 'S27',
                                          data=np.string_("")) 

        # TODO populate processingInformation/inputs
        inputs_ds = self.root.require_group("metadata/processingInformation/inputs")
        inputs_dataset_list = ["l0bGranules",
                               "orbitFiles",
                               "attitudeFiles",
                               "auxcalFiles",
                               "configFiles",
                               "demFiles"]

        for inp in inputs_dataset_list:
            if inp in g:
                continue
            inputs_ds.require_dataset(inp, (), 'S1', data=np.string_("")) 


    def swath(self, frequency="A") -> h5py.Group:
        return self.root.require_group(f"swaths/frequency{frequency}")

    def add_polarization(self, frequency="A", pol="HH"):
        assert len(pol) == 2 and pol[0] in "HVLR" and pol[1] in "HV"
        pols = np.string_([pol])  # careful not to promote to unicode below
        g = self.swath(frequency)
        name = "listOfPolarizations"
        if name in g:
            old_pols = np.array(g[name])
            assert pols[0] not in old_pols
            pols = np.append(old_pols, pols)
            del g[name]
        dset = g.create_dataset(name, data=pols)
        desc = f"List of polarization layers with frequency{frequency}"
        dset.attrs["description"] = np.string_(desc)

    def create_image(self, frequency="A", pol="HH", **kw) -> h5py.Dataset:
        log.info(f"Creating SLC image for frequency={frequency} pol={pol}")
        assert len(pol) == 2 and pol[0] in "HVLR" and pol[1] in "HV"
        self.add_polarization(frequency, pol)
        kw.setdefault("dtype", complex32)
        dset = self.swath(frequency).create_dataset(pol, **kw)
        dset.attrs["description"] = np.string_(f"Focused SLC image ({pol})")
        dset.attrs["units"] = np.string_("DN")
        return dset

    def update_swath(self, t: np.array, epoch: DateTime, r: np.array,
                     fc: float, frequency="A"):
        g = self.swath(frequency)
        # Time scale is in parent of group.  Use require_dataset to assert
        # matching time scale on repeated calls.
        d = g.parent.require_dataset("zeroDopplerTime", t.shape, t.dtype, data=t)
        d.attrs["units"] = np.string_(time_units(epoch))
        d.attrs["description"] = np.string_(
            "CF compliant dimension associated with azimuth time")

        d = g.parent.require_dataset("zeroDopplerTimeSpacing", (), float)
        d[()] = t[1] - t[0]
        d.attrs["units"] = np.string_("seconds")
        d.attrs["description"] = np.string_("Time interval in the along track"
            " direction for raster layers. This is same as the spacing between"
            " consecutive entries in the zeroDopplerTime array")

        d = g.require_dataset("slantRange", r.shape, r.dtype, data=r)
        d.attrs["units"] = np.string_("meters")
        d.attrs["description"] = np.string_("CF compliant dimension associated"
                                            " with slant range")

        d = g.require_dataset("slantRangeSpacing", (), float)
        d[()] = r[1] - r[0]
        d.attrs["units"] = np.string_("meters")
        d.attrs["description"] = np.string_("Slant range spacing of grid. Same"
            " as difference between consecutive samples in slantRange array")

        d = g.require_dataset("processedCenterFrequency", (), float)
        d[()] = fc
        d.attrs["units"] = np.string_("Hz")
        d.attrs["description"] = np.string_("Center frequency of the processed"
                                            " image in Hz")

        # TODO other parameters filled with bogus values for now, no units
        g.require_dataset("acquiredCenterFrequency", (), float)[()] = fc
        g.require_dataset("acquiredRangeBandwidth", (), float)[()] = 20e6
        g.require_dataset("nominalAcquisitionPRF", (), float)[()] = 1910.
        g.require_dataset("numberOfSubSwaths", (), int)[()] = 1
        g.require_dataset("processedAzimuthBandwidth", (), float)[()] = 1200.
        g.require_dataset("processedRangeBandwidth", (), float)[()] = 20e6
        g.require_dataset("sceneCenterAlongTrackSpacing", (), float)[()] = 4.
        g.require_dataset("sceneCenterGroundRangeSpacing", (), float)[()] = 12.
        d = g.require_dataset("validSamplesSubSwath1", (len(t), 2), 'int32')
        d[:] = (0, len(r))

    def set_orbit(self, orbit: Orbit, accel=None, type="Custom"):
        log.info("Writing orbit to SLC")
        g = self.root.require_group("metadata/orbit")
        orbit.save_to_h5(g)
        # interpMethod not in L1 spec. Delete it?
        # Add description attributes.  Should these go in saveToH5 method?
        g["time"].attrs["description"] = np.string_("Time vector record. This"
            " record contains the time corresponding to position, velocity,"
            " acceleration records")
        g["position"].attrs["description"] = np.string_("Position vector"
            " record. This record contains the platform position data with"
            " respect to WGS84 G1762 reference frame")
        g["velocity"].attrs["description"] = np.string_("Velocity vector"
            " record. This record contains the platform velocity data with"
            " respect to WGS84 G1762 reference frame")
        # Orbit source/type
        d = g.require_dataset("orbitType", (), "S10", data=np.string_(type))
        d.attrs["description"] = np.string_("PrOE (or) NOE (or) MOE (or) POE"
                                            " (or) Custom")
        # acceleration not stored in isce3 Orbit class.
        if accel is None:
            log.warning("Populating orbit/acceleration with zeros")
            accel = np.zeros_like(orbit.velocity)
        shape = orbit.velocity.shape
        if accel.shape != shape:
            raise ValueError("Acceleration dims must match orbit fields.")
        d = g.require_dataset("acceleration", shape, float, data=accel)
        d.attrs["description"] = np.string_("Acceleration vector record. This"
            " record contains the platform acceleration data with respect to"
            " WGS84 G1762 reference frame")
        d.attrs["units"] = np.string_("meters per second squared")

    def set_attitude(self, attitude: Attitude, epoch: DateTime, type="Custom"):
        log.info("Writing attitude to SLC")
        g = self.root.require_group("metadata/attitude")
        d = g.require_dataset("attitudeType", (), "S10", data=np.string_(type))
        d.attrs["description"] = np.string_("PrOE (or) NOE (or) MOE (or) POE"
                                            " (or) Custom")
        t = np.asarray(attitude.time)
        d = g.require_dataset("time", t.shape, t.dtype, data=t)
        d.attrs["description"] = np.string_("Time vector record. This record"
            " contains the time corresponding to attitude and quaternion"
            " records")
        d.attrs["units"] = np.string_(time_units(epoch))
        # TODO attitude rates
        n = len(attitude.time)
        qdot = np.zeros((n, 3))
        d = g.require_dataset("angularVelocity", (n,3), float, data=qdot)
        d.attrs["units"] = np.string_("radians per second")
        d.attrs["description"] = np.string_("Attitude angular velocity vectors"
                                            " (wx, wy, wz)")
        qv = np.array([[q.w, q.x, q.y, q.z] for q in attitude.quaternions])
        d = g.require_dataset("quaternions", qv.shape, qv.dtype, data=qv)
        d.attrs["units"] = np.string_("unitless")
        d.attrs["description"] = np.string_("Attitude quaternions"
                                            " (q0, q1, q2, q3)")
        rpy = np.asarray([[e.roll, e.pitch, e.yaw] for e in
            [EulerAngles(q) for q in attitude.quaternions]])
        d = g.require_dataset("eulerAngles", rpy.shape, rpy.dtype, data=rpy)
        d.attrs["units"] = np.string_("radians")
        d.attrs["description"] = np.string_("Attitude Euler angles"
                                            " (roll, pitch, yaw)")

    def copy_identification(self, raw: Raw, track: int = 0, frame: int = 0,
                            polygon: str = None, start_time: DateTime = None,
                            end_time: DateTime = None):
        """Copy the identification metadata from a L0B product.  Bounding
        polygon and start/end time will be updated if not None.
        """
        log.info(f"Populating identification based on {raw.filename}")
        # Most parameters are just copies of input ID.
        if self.idpath in self.root:
            del self.root[self.idpath]
        with h5py.File(raw.filename, 'r', libver='latest', swmr=True) as fd:
            self.root.copy(fd[raw.IdentificationPath], self.idpath)
        g = self.root[self.idpath]
        # Of course product type is different.
        d = set_string(g, "productType", self.product)
        d.attrs["description"] = np.string_("Product type")
        # L0B doesn't know about track/frame, so have to add it.
        d = g.require_dataset("trackNumber", (), 'uint8', data=track)
        d.attrs["units"] = np.string_("unitless")
        d.attrs["description"] = np.string_("Track number")
        d = g.require_dataset("frameNumber", (), 'uint16', data=frame)
        d.attrs["units"] = np.string_("unitless")
        d.attrs["description"] = np.string_("Frame number")
        # Polygon different due to reskew and possibly multiple input L0Bs.
        if polygon is not None:
            d = set_string(g, "boundingPolygon", polygon)
            d.attrs["epsg"] = 4326
            d.attrs["description"] = np.string_("OGR compatible WKT"
                " representation of bounding polygon of the image")
            d.attrs["ogr_geometry"] = np.string_("polygon")
        else:
            log.warning("SLC bounding polygon not updated.  Using L0B polygon.")
        # Start/end time can be customized via runconfig and generally are
        # different anyway due to reskew.
        if start_time is not None:
            d = set_string(g, "zeroDopplerStartTime", start_time.isoformat())
            d.attrs["description"] = np.string_("Azimuth start time of product")
        else:
            log.warning("SLC start time not updated.  Using L0B start time.")
        if end_time is not None:
            d = set_string(g, "zeroDopplerEndTime", end_time.isoformat())
            d.attrs["description"] = np.string_("Azimuth stop time of product")
        else:
            log.warning("SLC end time not updated.  Using L0B end time.")


    def set_geolocation_grid(self, orbit: Orbit, grid: RadarGridParameters,
                             doppler: LUT2d, epsg=4326, dem=DEMInterpolator(),
                             threshold=1e-8, maxiter=50, delta_range=10.0):
        log.info(f"Creating geolocationGrid.")
        # TODO Get DEM stats.  Until then just span all Earthly values.
        heights = np.linspace(-500, 9000, 20)
        # Figure out decimation factors that give < 500 m spacing.
        max_spacing = 500.
        t = (grid.sensing_mid +
             (grid.ref_epoch - orbit.reference_epoch).total_seconds())
        _, v = orbit.interpolate(t)
        dx = np.linalg.norm(v) / grid.prf
        tskip = int(np.floor(max_spacing / dx))
        rskip = int(np.floor(max_spacing / grid.range_pixel_spacing))
        grid = grid[::tskip, ::rskip]

        group_name = f"{self.root.name}/metadata/geolocationGrid"
        rslc_doppler = LUT2d()  # RSLCs are zero-Doppler by definition
        # Change spelling of geo2rdr params
        tol = dict(
            threshold_geo2rdr = threshold,
            numiter_geo2rdr = maxiter,
            delta_range = delta_range,
        )
        add_geolocation_grid_cubes_to_hdf5(self, group_name, grid, heights,
            orbit, doppler, rslc_doppler, epsg, **tol)


    def add_calibration_section(self, frequency, pol, 
                                az_time_orig_vect: np.array, 
                                epoch: DateTime, 
                                slant_range_orig_vect: np.array):
        assert len(pol) == 2 and pol[0] in "HVLR" and pol[1] in "HV"

        calibration_section_sampling = 50
        g = self.root.require_group("metadata/calibrationInformation")

        if "zeroDopplerTime" in g:
            t = g['zeroDopplerTime']
        else:
            t = az_time_orig_vect[0:-1:calibration_section_sampling]
            d = g.require_dataset("zeroDopplerTime", t.shape, t.dtype, data=t)
            d.attrs["units"] = np.string_(time_units(epoch))
            d.attrs["description"] = np.string_(
                "CF compliant dimension associated with azimuth time")

        if "slantRange" in g:
            r = g['slantRange']
        else:
            r = slant_range_orig_vect[0:-1:calibration_section_sampling]
            d = g.require_dataset("slantRange", r.shape, r.dtype, data=r)
            d.attrs["units"] = np.string_("meters")
            d.attrs["description"] = np.string_("CF compliant dimension associated"
                                                " with slant range")

        dummy_array = np.ones((t.size, r.size), dtype=np.float32)

        if "geometry/beta0" not in g:
            d = g.require_dataset(f"geometry/beta0", dummy_array.shape, 
                                  np.float32, data=dummy_array)
            d.attrs["description"] = np.string_(
                "2D LUT to convert DN to beta 0 assuming as a function"
                 " of zero doppler time and slant range")

        if "geometry/sigma0" not in g:
            d = g.require_dataset(f"geometry/sigma0", dummy_array.shape, 
                                  np.float32, data=dummy_array)
            d.attrs["description"] = np.string_(
                "2D LUT to convert DN to sigma 0 assuming as a function"
                 " of zero doppler time and slant range")


        if "geometry/gamma0" not in g:
            d = g.require_dataset(f"geometry/gamma0", dummy_array.shape, 
                                  np.float32, data=dummy_array)
            d.attrs["description"] = np.string_(
                "2D LUT to convert DN to gamma 0 as a function of zero"
                " doppler time and slant range")

        d = g.require_dataset(
            f"frequency{frequency}/{pol}/elevationAntennaPattern", 
            dummy_array.shape, np.float32, data=dummy_array)
        d.attrs["description"] = np.string_(
            "Complex two-way elevation antenna pattern")

        dummy_array = np.zeros((t.size, r.size))
        d = g.require_dataset(
            f"frequency{frequency}/{pol}/nes0", 
            dummy_array.shape, np.float32, data=dummy_array)
        d.attrs["description"] = np.string_(
            "Thermal noise equivalent sigma0")
