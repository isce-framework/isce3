import h5py
import logging
import numpy as np
from pybind_isce3.core import LUT2d, DateTime, Orbit, Quaternion
from pybind_isce3.product import RadarGridParameters
from pybind_nisar.h5 import set_string
from pybind_nisar.types import complex32
from pybind_nisar.products.readers.Raw import Raw

log = logging.getLogger("SLCWriter")

# TODO refactor isce::io::setRefEpoch
def time_units(epoch: DateTime) -> str:
    # XXX isce::io::*RefEpoch don't parse or serialize fraction.
    if epoch.frac != 0.0:
        raise ValueError("Reference epoch must have integer seconds.")
    date = "{:04d}-{:02d}-{:02d}".format(epoch.year, epoch.month, epoch.day)
    time = "{:02d}:{:02d}:{:02d}".format(epoch.hour, epoch.minute, epoch.second)
    return "seconds since " + date + " " + time


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
        dop.save_to_h5(g, f"{name}/dopplerCentroid", epoch, "Hz")
        # TODO veff, fmrate not used anywhere afaict except product io.
        v = np.zeros_like(dop.data)
        g.require_dataset("effectiveVelocity", v.shape, v.dtype, data=v)
        fg.require_dataset("azimuthFMRate", v.shape, v.dtype, data=v)
        # TODO weighting, ref height

    def swath(self, frequency="A") -> h5py.Group:
        return self.root.require_group(f"swaths/frequency{frequency}")

    def add_polarization(self, frequency="A", pol="HH"):
        assert len(pol) == 2 and pol[0] in "HVLR" and pol[1] in "HV"
        g = self.swath(frequency)
        name = "listOfPolarizations"
        if name in g:
            pols = np.array(g[name])
            assert(pol not in pols)
            pols = np.append(pols, [pol])
            del g[name]
        else:
            pols = np.array([pol], dtype="S2")
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

    def set_attitude(self, attitude: Quaternion, epoch: DateTime, type="Custom"):
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
        q = attitude.qvec
        d = g.require_dataset("quaternions", q.shape, q.dtype, data=q)
        d.attrs["units"] = np.string_("unitless")
        d.attrs["description"] = np.string_("Attitude quaternions"
                                            " (q0, q1, q2, q3)")
        ypr = np.array([attitude.ypr(t) for t in attitude.time])
        rpy = ypr[:,::-1]
        d = g.require_dataset("eulerAngles", rpy.shape, rpy.dtype, data=rpy)
        d.attrs["units"] = np.string_("radians")
        d.attrs["description"] = np.string_("Attitude Euler angles"
                                            " (roll, pitch, yaw)")

    def copy_identification(self, raw: Raw, track: int = 0, frame: int = 0,
                            polygon: str = None):
        """Copy the identification metadata from a L0B product.  Bounding
        polygon will be updated if not None.
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
